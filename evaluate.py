import csv

identical_input = 0
identical_output = 0
diff_input = 0
diff_output = 0
total = 0
empty_pz = 0
input_diffs = []
output_diffs = []

with open('/Users/huybro/Desktop/lotus-experiment/filter_fever.csv', 'r') as f:
    reader = csv.DictReader(f)
    cols = list(next(csv.DictReader(open('/Users/huybro/Desktop/lotus-experiment/filter_fever.csv'))).keys())
    print(f'Columns: {cols}')
    
    for i, row in enumerate(reader):
        total += 1
        li = row.get('lotus_input', '')
        pi = row.get('pz_input', '')
        lo = row.get('lotus_output', '')
        po = row.get('pz_output', '')
        
        if not pi.strip():
            empty_pz += 1
            continue
        
        if li == pi:
            identical_input += 1
        else:
            diff_input += 1
            if len(input_diffs) < 3:
                for j, (a, b) in enumerate(zip(li, pi)):
                    if a != b:
                        input_diffs.append({'tuple': i, 'claim': row.get('claim','')[:60], 'pos': j, 'lotus': repr(li[max(0,j-20):j+20]), 'pz': repr(pi[max(0,j-20):j+20])})
                        break
        
        if lo == po:
            identical_output += 1
        else:
            diff_output += 1
            if len(output_diffs) < 3:
                output_diffs.append({'tuple': i, 'claim': row.get('claim','')[:60], 'lotus_len': len(lo), 'pz_len': len(po)})

compared = total - empty_pz
print(f'\nTotal tuples: {total}')
print(f'Empty PZ: {empty_pz}')
print(f'Compared: {compared}')
print()
if compared > 0:
    print(f'=== INPUTS ===')
    print(f'  Identical: {identical_input}/{compared} ({100*identical_input/compared:.1f}%)')
    print(f'  Different: {diff_input}/{compared} ({100*diff_input/compared:.1f}%)')
    print()
    print(f'=== OUTPUTS ===')
    print(f'  Identical: {identical_output}/{compared} ({100*identical_output/compared:.1f}%)')
    print(f'  Different: {diff_output}/{compared} ({100*diff_output/compared:.1f}%)')

if input_diffs:
    print()
    print(f'=== SAMPLE INPUT DIFFS ===')
    for d in input_diffs:
        print(f"  Tuple {d['tuple']}: {d['claim']}")
        print(f"    LOTUS: {d['lotus']}")
        print(f"    PZ:    {d['pz']}")

if output_diffs:
    print()
    print(f'=== SAMPLE OUTPUT DIFFS ===')
    for d in output_diffs:
        print(f"  Tuple {d['tuple']}: {d['claim']} (L={d['lotus_len']}, PZ={d['pz_len']} chars)")

