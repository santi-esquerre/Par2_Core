#!/bin/bash
# Run all 12 test cases: 6 YAMLs × 2 velocity fields (field0, field1)
# Uses compare_original_vs_core.py for proper comparison, then adds Core-only analysis

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
YAML_DIR="$SCRIPT_DIR/legacy_runner/legacy_format_examples"
BUILD_DIR="$BASE_DIR/build"
WORKDIR="$BASE_DIR/debug_artifacts"

PAR2_ORIG="$BUILD_DIR/par2_original"
PAR2_CORE="$BUILD_DIR/tools/legacy_runner/par2core_legacy_runner"

COMPARE_SCRIPT="$SCRIPT_DIR/compare_original_vs_core.py"

YAMLS=(
    test01_baseline_trilinear
    test02_baseline_finite_difference
    test03_advection_only
    test04_diffusion_only
    test05_high_dispersivity
    test06_injection_near_boundary
)

FIELDS=(field0 field1)

PASS_COUNT=0
FAIL_COUNT=0
ERROR_COUNT=0
declare -A RESULTS

for field in "${FIELDS[@]}"; do
    for yaml_name in "${YAMLS[@]}"; do
        CASE="${yaml_name}/${field}"
        CASE_DIR="$WORKDIR/${yaml_name}/${field}"
        mkdir -p "$CASE_DIR"
        
        echo ""
        echo "================================================================"
        echo "  CASE: ${yaml_name} / ${field}"
        echo "================================================================"
        
        # Create modified YAML that references the correct FTL
        ORIG_YAML="$YAML_DIR/${yaml_name}.yaml"
        CASE_YAML="$CASE_DIR/${yaml_name}.yaml"
        
        # Replace velocity file reference: example.ftl -> fieldX.ftl
        sed "s|file: example.ftl|file: ${field}.ftl|g" "$ORIG_YAML" > "$CASE_YAML"
        
        # Copy the correct FTL file to case dir
        cp "$YAML_DIR/${field}.ftl" "$CASE_DIR/${field}.ftl"
        
        # Run comparison
        python3 "$COMPARE_SCRIPT" \
            --yaml "$CASE_YAML" \
            --par2_original "$PAR2_ORIG" \
            --par2_core "$PAR2_CORE" \
            --workdir "$CASE_DIR" \
            --keep-workdir \
            --stat-rtol 0.25 \
            -v 2>&1 | tee "$CASE_DIR/comparison.log"
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        if [ $EXIT_CODE -eq 0 ]; then
            RESULTS["$CASE"]="PASS"
            PASS_COUNT=$((PASS_COUNT + 1))
        elif [ $EXIT_CODE -eq 1 ]; then
            RESULTS["$CASE"]="FAIL (comparison mismatch)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            RESULTS["$CASE"]="ERROR (execution error)"
            ERROR_COUNT=$((ERROR_COUNT + 1))
        fi

        # --- Core-only analysis (independent of Legacy) ---
        echo ""
        echo "  --- Core-Only Analysis ---"
        # Find last snapshot from core
        CORE_DIR="$CASE_DIR/core_case"
        if [ -d "$CORE_DIR" ]; then
            LAST_SNAP=$(find "$CORE_DIR" -name "*snap*" | sort | tail -1)
            if [ -n "$LAST_SNAP" ]; then
                python3 -c "
import numpy as np

d = np.loadtxt('$LAST_SNAP', delimiter=',', skiprows=1, usecols=(1,2,3))
nans = np.sum(np.isnan(d))
n = d.shape[0]
print(f'  Particles: {n}')
print(f'  NaN values: {nans}')
for ax, name in enumerate(['X','Y','Z']):
    vals = d[:,ax]
    print(f'  {name}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}')
# Check domain bounds [0, 100]
oob = np.sum((d < 0) | (d > 100))
print(f'  Out-of-bounds [0,100]: {oob}')
# Check frozen particles (std=0 on any axis)
for ax, name in enumerate(['X','Y','Z']):
    if np.std(d[:,ax]) < 1e-10:
        print(f'  WARNING: All particles frozen on {name} axis (std≈0)')
" 2>&1
            fi
        fi
    done
done

echo ""
echo "================================================================"
echo "  FINAL SUMMARY"
echo "================================================================"
printf "  %-50s %s\n" "CASE" "RESULT"
printf "  %-50s %s\n" "----" "------"
for field in "${FIELDS[@]}"; do
    for yaml_name in "${YAMLS[@]}"; do
        CASE="${yaml_name}/${field}"
        printf "  %-50s %s\n" "$CASE" "${RESULTS[$CASE]}"
    done
done
echo ""
echo "  Total: ${PASS_COUNT} PASS, ${FAIL_COUNT} FAIL, ${ERROR_COUNT} ERROR"
echo "================================================================"
