make 
for ((power = 10; power <= 16; power++)); do
    argument=$((2**power))
    argumentwo=$((3*$argument))
    ./managed $argument $argument $argument 2
    ./manual $argument $argument $argument 2
    ./async $argument 2
    # echo -e "\n\n"
done
