#!/usr/bin/env bash
# run_norm_eval.sh — Run RMSNorm and LayerNorm benchmarks across HBM configs
#                     and vector lengths, then generate LaTeX tables and CSVs.
#
# Usage:
#   bash scripts/run_norm_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
PIMEVAL_DIR="${REPO_ROOT}/PIMeval-PIMbench"
CONFIG_DIR="${PIMEVAL_DIR}/configs/hbm"

BENCHMARKS=(rmsnorm lnorm)
VECTOR_LENGTHS=(128 4096 8192 16384)
CONFIGS=(
    PIMeval_Bank_Rank1.cfg
    PIMeval_Bank_Rank4.cfg
    PIMeval_Bank_Rank8.cfg
    PIMeval_Bank_Rank16.cfg
    PIMeval_Bank_Rank32.cfg
)

# ---------- Helper: parse PIM output ----------
parse_pim_output() {
    local output="$1"

    local dc_runtime dc_energy
    dc_runtime=$(echo "$output" | grep -A5 "Data Copy Stats:" | grep "TOTAL" \
        | sed 's/.*TOTAL ---------.*bytes[[:space:]]*//' | awk '{print $1}')
    dc_energy=$(echo "$output" | grep -A5 "Data Copy Stats:" | grep "TOTAL" \
        | sed 's/.*Estimated Runtime[[:space:]]*//' | awk '{print $1}')

    local pim_runtime pim_energy
    pim_runtime=$(echo "$output" | grep -A20 "PIM Command Stats:" | grep "TOTAL" \
        | awk -F':' '{print $2}' | awk '{print $2}')
    pim_energy=$(echo "$output" | grep -A20 "PIM Command Stats:" | grep "TOTAL" \
        | awk -F':' '{print $2}' | awk '{print $3}')

    local host_time
    host_time=$(echo "$output" | grep -i "Host elapsed time" | grep -oP '[\d.]+(?=\s*ms)')

    dc_runtime=${dc_runtime:-0}
    dc_energy=${dc_energy:-0}
    pim_runtime=${pim_runtime:-0}
    pim_energy=${pim_energy:-0}
    host_time=${host_time:-0}

    echo "${dc_runtime} ${dc_energy} ${pim_runtime} ${pim_energy} ${host_time}"
}

# ---------- Run each benchmark ----------
for bench in "${BENCHMARKS[@]}"; do
    PIM_DIR="${REPO_ROOT}/${bench}/PIM"
    CPU_DIR="${REPO_ROOT}/${bench}/baselines/CPU"
    OUTPUT_DIR="${REPO_ROOT}/output/${bench}-sweep"
    LOG_DIR="${OUTPUT_DIR}/logs"
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

    RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    echo "${bench^^} Evaluation — $(date)" | tee "${RUN_LOG}"
    echo "Vector lengths: ${VECTOR_LENGTHS[*]}" | tee -a "${RUN_LOG}"
    echo "HBM configs: ${CONFIGS[*]}" | tee -a "${RUN_LOG}"
    echo "========================================" | tee -a "${RUN_LOG}"

    # -- Associative arrays: key = "rank_veclen" --
    declare -A dc_times dc_energies cmd_times cmd_energies host_times total_times total_energies
    declare -A cpu_times

    # ---------- Run CPU baseline for each vector length ----------
    for vlen in "${VECTOR_LENGTHS[@]}"; do
        echo "==> [${bench}] CPU baseline, veclen=${vlen}..." | tee -a "${RUN_LOG}"
        cpu_output=$(cd "${CPU_DIR}" && ./${bench}.out -l ${vlen} 2>&1) || true
        echo "$cpu_output" | tee -a "${RUN_LOG}"
        echo "$cpu_output" > "${LOG_DIR}/cpu_l${vlen}.log"

        ct=$(echo "$cpu_output" | grep -i "Duration" | grep -oP '[\d.]+(?=\s*ms)')
        cpu_times["${vlen}"]=${ct:-N/A}
        echo "    CPU Duration: ${cpu_times[${vlen}]} ms" | tee -a "${RUN_LOG}"
        echo "" | tee -a "${RUN_LOG}"
    done

    # ---------- Run PIM configs x vector lengths ----------
    for cfg in "${CONFIGS[@]}"; do
        rank=$(echo "$cfg" | grep -oP 'Rank\K[0-9]+')
        config_path="${CONFIG_DIR}/${cfg}"

        for vlen in "${VECTOR_LENGTHS[@]}"; do
            key="${rank}_${vlen}"
            echo "==> [${bench}] Rank ${rank}, veclen=${vlen}..." | tee -a "${RUN_LOG}"

            pim_output=$(cd "${PIM_DIR}" && ./${bench}.out \
                -l ${vlen} \
                -c "${config_path}" \
                -v t 2>&1) || true
            echo "$pim_output" | tee -a "${RUN_LOG}"
            echo "$pim_output" > "${LOG_DIR}/pim_rank${rank}_l${vlen}.log"

            read -r dc_rt dc_en pim_rt pim_en h_time <<< "$(parse_pim_output "$pim_output")"

            dc_times["$key"]="$dc_rt"
            dc_energies["$key"]="$dc_en"
            cmd_times["$key"]="$pim_rt"
            cmd_energies["$key"]="$pim_en"
            host_times["$key"]="$h_time"
            total_times["$key"]=$(echo "${dc_rt} + ${pim_rt} + ${h_time}" | bc -l)
            total_energies["$key"]=$(echo "${dc_en} + ${pim_en}" | bc -l)

            echo "    Total Time = ${total_times[$key]} ms, Total Energy = ${total_energies[$key]} mJ" | tee -a "${RUN_LOG}"
            echo "" | tee -a "${RUN_LOG}"
        done
    done

    # ---------- Generate CSV: detailed results ----------
    CSV_FILE="${OUTPUT_DIR}/${bench}_results.csv"
    echo "Configuration,Vector Length,Data Copy (ms),PIM Compute (ms),Host (ms),Total PIM Time (ms),Total PIM Energy (mJ),CPU Time (ms)" > "${CSV_FILE}"

    for cfg in "${CONFIGS[@]}"; do
        rank=$(echo "$cfg" | grep -oP 'Rank\K[0-9]+')
        for vlen in "${VECTOR_LENGTHS[@]}"; do
            key="${rank}_${vlen}"
            echo "Rank ${rank},${vlen},${dc_times[$key]},${cmd_times[$key]},${host_times[$key]},${total_times[$key]},${total_energies[$key]},${cpu_times[$vlen]}" >> "${CSV_FILE}"
        done
    done

    # ---------- Generate LaTeX table: Time ----------
    TEX_TIME="${OUTPUT_DIR}/${bench}_time_table.tex"
    BENCH_UPPER="${bench^^}"
    cat > "${TEX_TIME}" << TEXHEADER
% ${BENCH_UPPER} Time Comparison
% Requires: booktabs, siunitx packages
\\begin{table}[t]
\\centering
\\caption{${BENCH_UPPER} total execution time (ms) across HBM Bank-Level PIM configurations and CPU baseline.}
\\label{tab:${bench}-time}
\\begin{tabular}{lSSSS}
\\toprule
{Configuration} & {\$n=128\$} & {\$n=4096\$} & {\$n=8192\$} & {\$n=16384\$} \\\\
\\midrule
TEXHEADER

    for cfg in "${CONFIGS[@]}"; do
        rank=$(echo "$cfg" | grep -oP 'Rank\K[0-9]+')
        printf 'Rank %-2s' "${rank}" >> "${TEX_TIME}"
        for vlen in "${VECTOR_LENGTHS[@]}"; do
            key="${rank}_${vlen}"
            printf ' & %s' "${total_times[$key]}" >> "${TEX_TIME}"
        done
        printf ' \\\\\n' >> "${TEX_TIME}"
    done

    # CPU baseline row
    printf '\\midrule\nCPU Baseline' >> "${TEX_TIME}"
    for vlen in "${VECTOR_LENGTHS[@]}"; do
        printf ' & %s' "${cpu_times[$vlen]}" >> "${TEX_TIME}"
    done
    printf ' \\\\\n' >> "${TEX_TIME}"

    cat >> "${TEX_TIME}" << 'TEXFOOTER'
\bottomrule
\end{tabular}
\end{table}
TEXFOOTER

    # ---------- Generate LaTeX table: Energy ----------
    TEX_ENERGY="${OUTPUT_DIR}/${bench}_energy_table.tex"
    cat > "${TEX_ENERGY}" << TEXHEADER
% ${BENCH_UPPER} Energy Consumption
% Requires: booktabs, siunitx packages
\\begin{table}[t]
\\centering
\\caption{${BENCH_UPPER} total energy consumption (mJ) across HBM Bank-Level PIM configurations.}
\\label{tab:${bench}-energy}
\\begin{tabular}{lSSSS}
\\toprule
{Configuration} & {\$n=128\$} & {\$n=4096\$} & {\$n=8192\$} & {\$n=16384\$} \\\\
\\midrule
TEXHEADER

    for cfg in "${CONFIGS[@]}"; do
        rank=$(echo "$cfg" | grep -oP 'Rank\K[0-9]+')
        printf 'Rank %-2s' "${rank}" >> "${TEX_ENERGY}"
        for vlen in "${VECTOR_LENGTHS[@]}"; do
            key="${rank}_${vlen}"
            printf ' & %s' "${total_energies[$key]}" >> "${TEX_ENERGY}"
        done
        printf ' \\\\\n' >> "${TEX_ENERGY}"
    done

    cat >> "${TEX_ENERGY}" << 'TEXFOOTER'
\bottomrule
\end{tabular}
\end{table}
TEXFOOTER

    echo "============================================" | tee -a "${RUN_LOG}"
    echo "[${bench}] Done. Output:" | tee -a "${RUN_LOG}"
    echo "  LaTeX: ${TEX_TIME}" | tee -a "${RUN_LOG}"
    echo "         ${TEX_ENERGY}" | tee -a "${RUN_LOG}"
    echo "  CSV:   ${CSV_FILE}" | tee -a "${RUN_LOG}"
    echo "  Logs:  ${LOG_DIR}/" | tee -a "${RUN_LOG}"
    echo "============================================" | tee -a "${RUN_LOG}"

    # Clean up associative arrays for next benchmark
    unset dc_times dc_energies cmd_times cmd_energies host_times total_times total_energies cpu_times
done
