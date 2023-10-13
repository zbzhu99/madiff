for map in 3m 5m_vs_6m 2s3z
do
    for quality in Good Medium Poor
    do
        echo "Copying ./reference_repos/og_marl/datasets/smac/$map/$quality/*.npy to ./diffuser/datasets/data/smac/$map/$quality/"
        cp ./reference_repos/og_marl/datasets/smac/$map/$quality/*.npy ./diffuser/datasets/data/smac/$map/$quality/
    done
done