for config_file in configs/*.yaml; do
    echo "Running $config_file"
    python cnntest.py --config "$config_file"

done
