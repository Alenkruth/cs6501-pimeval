#!/usr/bin/env bash
# run_gemv_eval.sh — Run GEMV benchmarks across HBM Bank-Level PIM configs and CPU baseline,
#                     then generate LaTeX tables with results.
#
# Usage:
#   bash run_gemv_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
PIMEVAL_DIR="${REPO_ROOT}/PIMeval-PIMbench"
GEMV_PIM_DIR="${PIMEVAL_DIR}/PIMbench/gemv/PIM"
GEMV_CPU_DIR="${PIMEVAL_DIR}/PIMbench/gemv/baselines/CPU"
CONFIG_DIR="${PIMEVAL_DIR}/configs/hbm"
OUTPUT_DIR="${REPO_ROOT}/output/gemv-sweep"

MATRIX_ROWS=4096
MATRIX_COLS=4096

# Configs to evaluate (sorted by rank count)
CONFIGS=(
    PIMeval_Bank_Rank1.cfg
    PIMeval_Bank_Rank4.cfg
    PIMeval_Bank_Rank8.cfg
    PIMeval_Bank_Rank16.cfg
    PIMeval_Bank_Rank32.cfg
)

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Combined log file with timestamp
RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "GEMV Evaluation — $(date)" | tee "${RUN_LOG}"
echo "Matrix: ${MATRIX_ROWS}x${MATRIX_COLS}" | tee -a "${RUN_LOG}"
echo "========================================" | tee -a "${RUN_LOG}"

# ---------- Helper: parse PIM output ----------
# Extracts data-copy runtime/energy and PIM-command runtime/energy from stdout.
# Prints: <data_copy_ms> <data_copy_mj> <pim_cmd_ms> <pim_cmd_mj>
parse_pim_output() {
    local output="$1"

    # Data Copy TOTAL line format:
    #   TOTAL --------- : <bytes> bytes   <runtime> ms Estimated Runtime   <energy> mj Estimated Energy
    local dc_runtime dc_energy
    dc_runtime=$(echo "$output" | grep -A1 "Data Copy Stats:" | grep "TOTAL" \
        | sed 's/.*TOTAL ---------.*bytes[[:space:]]*//' | awk '{print $1}')
    dc_energy=$(echo "$output" | grep -A1 "Data Copy Stats:" | grep "TOTAL" \
        | sed 's/.*Estimated Runtime[[:space:]]*//' | awk '{print $1}')

    # PIM Command TOTAL line format:
    #   TOTAL --------- : <count> <runtime> <energy> <gops_w> <%R> <%W> <%L>
    local pim_runtime pim_energy
    pim_runtime=$(echo "$output" | grep -A20 "PIM Command Stats:" | grep "TOTAL" \
        | awk -F':' '{print $2}' | awk '{print $2}')
    pim_energy=$(echo "$output" | grep -A20 "PIM Command Stats:" | grep "TOTAL" \
        | awk -F':' '{print $2}' | awk '{print $3}')

    # Default to 0 if parsing fails
    dc_runtime=${dc_runtime:-0}
    dc_energy=${dc_energy:-0}
    pim_runtime=${pim_runtime:-0}
    pim_energy=${pim_energy:-0}

    echo "${dc_runtime} ${dc_energy} ${pim_runtime} ${pim_energy}"
}

# ---------- Run CPU baseline ----------
echo "==> Running CPU baseline (${MATRIX_ROWS}x${MATRIX_COLS})..." | tee -a "${RUN_LOG}"
cpu_output=$(cd "${GEMV_CPU_DIR}" && ./gemv.out -r ${MATRIX_ROWS} -c ${MATRIX_COLS} 2>&1) || true
echo "$cpu_output" | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"

# Save individual log
echo "$cpu_output" > "${LOG_DIR}/cpu_baseline.log"

cpu_time=$(echo "$cpu_output" | grep -i "Duration" | grep -oP '[\d.]+(?=\s*ms)')
cpu_time=${cpu_time:-N/A}
echo "    CPU Duration: ${cpu_time} ms" | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"

# ---------- Run PIM configs ----------
declare -a pim_ranks
declare -a pim_total_times
declare -a pim_total_energies
declare -a pim_dc_times
declare -a pim_dc_energies
declare -a pim_cmd_times
declare -a pim_cmd_energies

for cfg in "${CONFIGS[@]}"; do
    rank=$(echo "$cfg" | grep -oP 'Rank\K[0-9]+')
    config_path="${CONFIG_DIR}/${cfg}"

    echo "==> Running PIM with ${cfg} (Rank ${rank})..." | tee -a "${RUN_LOG}"

    pim_output=$(cd "${GEMV_PIM_DIR}" && ./gemv.out \
        -r ${MATRIX_ROWS} -d ${MATRIX_COLS} \
        -c "${config_path}" \
        -v t 2>&1) || true
    echo "$pim_output" | tee -a "${RUN_LOG}"
    echo "" | tee -a "${RUN_LOG}"

    # Save individual log
    echo "$pim_output" > "${LOG_DIR}/pim_rank${rank}.log"

    read -r dc_rt dc_en pim_rt pim_en <<< "$(parse_pim_output "$pim_output")"

    total_time=$(echo "${dc_rt} + ${pim_rt}" | bc -l)
    total_energy=$(echo "${dc_en} + ${pim_en}" | bc -l)

    pim_ranks+=("${rank}")
    pim_dc_times+=("${dc_rt}")
    pim_dc_energies+=("${dc_en}")
    pim_cmd_times+=("${pim_rt}")
    pim_cmd_energies+=("${pim_en}")
    pim_total_times+=("${total_time}")
    pim_total_energies+=("${total_energy}")

    echo "    Rank ${rank}: Total Time = ${total_time} ms, Total Energy = ${total_energy} mJ" | tee -a "${RUN_LOG}"
    echo "" | tee -a "${RUN_LOG}"
done

# ---------- Generate LaTeX table: Time comparison ----------
cat > "${OUTPUT_DIR}/gemv_time_table.tex" << 'TEXHEADER'
% GEMV Time Comparison (4096x4096 matrix, 4096x1 vector)
% Requires: booktabs, siunitx packages
\begin{table}[t]
\centering
\caption{GEMV execution time across HBM Bank-Level PIM configurations and CPU baseline (matrix: $4096 \times 4096$, vector: $4096 \times 1$).}
\label{tab:gemv-time}
\begin{tabular}{lSSS}
\toprule
{Configuration} & {Data Copy (ms)} & {PIM Compute (ms)} & {Total Time (ms)} \\
\midrule
TEXHEADER

for i in "${!pim_ranks[@]}"; do
    printf 'Rank %-2s & %s & %s & %s \\\\\n' \
        "${pim_ranks[$i]}" \
        "${pim_dc_times[$i]}" \
        "${pim_cmd_times[$i]}" \
        "${pim_total_times[$i]}" >> "${OUTPUT_DIR}/gemv_time_table.tex"
done

cat >> "${OUTPUT_DIR}/gemv_time_table.tex" << TEXFOOTER
\\midrule
CPU Baseline & {--} & {--} & ${cpu_time} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
TEXFOOTER

# ---------- Generate LaTeX table: Energy ----------
cat > "${OUTPUT_DIR}/gemv_energy_table.tex" << 'TEXHEADER'
% GEMV Energy Consumption (4096x4096 matrix, 4096x1 vector)
% Requires: booktabs, siunitx packages
\begin{table}[t]
\centering
\caption{GEMV energy consumption across HBM Bank-Level PIM configurations (matrix: $4096 \times 4096$, vector: $4096 \times 1$).}
\label{tab:gemv-energy}
\begin{tabular}{lSSS}
\toprule
{Configuration} & {Data Copy (mJ)} & {PIM Compute (mJ)} & {Total Energy (mJ)} \\
\midrule
TEXHEADER

for i in "${!pim_ranks[@]}"; do
    printf 'Rank %-2s & %s & %s & %s \\\\\n' \
        "${pim_ranks[$i]}" \
        "${pim_dc_energies[$i]}" \
        "${pim_cmd_energies[$i]}" \
        "${pim_total_energies[$i]}" >> "${OUTPUT_DIR}/gemv_energy_table.tex"
done

cat >> "${OUTPUT_DIR}/gemv_energy_table.tex" << 'TEXFOOTER'
\bottomrule
\end{tabular}
\end{table}
TEXFOOTER

# ---------- Generate CSV ----------
CSV_FILE="${OUTPUT_DIR}/gemv_results.csv"
echo "Configuration,Data Copy (ms),PIM Compute (ms),Total PIM Time (ms),Data Copy (mJ),PIM Compute (mJ),Total PIM Energy (mJ),CPU Time (ms)" > "${CSV_FILE}"

for i in "${!pim_ranks[@]}"; do
    echo "Rank ${pim_ranks[$i]},${pim_dc_times[$i]},${pim_cmd_times[$i]},${pim_total_times[$i]},${pim_dc_energies[$i]},${pim_cmd_energies[$i]},${pim_total_energies[$i]},${cpu_time}" >> "${CSV_FILE}"
done

echo "============================================" | tee -a "${RUN_LOG}"
echo "Done. Output written to:" | tee -a "${RUN_LOG}"
echo "  LaTeX: ${OUTPUT_DIR}/gemv_time_table.tex" | tee -a "${RUN_LOG}"
echo "         ${OUTPUT_DIR}/gemv_energy_table.tex" | tee -a "${RUN_LOG}"
echo "  CSV:   ${CSV_FILE}" | tee -a "${RUN_LOG}"
echo "  Logs:  ${LOG_DIR}/" | tee -a "${RUN_LOG}"
echo "============================================" | tee -a "${RUN_LOG}"
