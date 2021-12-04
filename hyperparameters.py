import os

#features = os.listdir('working_directory')

for lr in [1e-1,1e-2]:
    for atom_fea_len in [64,128]:
        for p_fea_len in [18, 36, 72]:
            for n_h in [3,5,7]:
                if os.path.exists('results_%s_%d_%d_%d.csv'%(str(lr),atom_fea_len,p_fea_len,n_h)):
                    continue
                os.system('NV_GPU=0 nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mf/missing_values/fine_tuning nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 main_co_cgcnn.py --lr %f --epochs 200 --atom-fea-len %d --h-fea-len %d --p-fea-len %d --n-h %d /home/shengg/mf/whole_dataset/training/sample'%(lr,atom_fea_len,atom_fea_len,p_fea_len,n_h))
                os.system('rm -f test_results.csv')
                os.system('NV_GPU=1 nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mf/missing_values/fine_tuning nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 predict_co_cgcnn.py model_best.pth.tar /home/shengg/mf/whole_dataset/test/sample')
                os.system('mv -f model_best.pth.tar model_%s_%d_%d_%d.pth.tar'%(str(lr),atom_fea_len,p_fea_len,n_h))
                os.system('mv -f test_results.csv results_%s_%d_%d_%d.csv'%(str(lr),atom_fea_len,p_fea_len,n_h))
