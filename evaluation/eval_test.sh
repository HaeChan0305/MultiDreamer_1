# Example iteration code for evaluation
# make sure that activate the conda environment before execute

my_input=(1 4 6 9 10 12 13 14 15 18 19 21 22 23 25 29 31 33 35 36 37 38 39 40 41 42 43 44 46 48 49 50)

for NUM in "${my_input[@]}"; do
  python eval.py --dir "../data/eval" --input $NUM --filename "result"
done

