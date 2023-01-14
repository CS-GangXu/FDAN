# FDAN
The official PyTorch implementation for FDAN.

# Dataset
You can download our SRITM-4K test set from [Baidu Netdisk (code: fhyt)](https://pan.baidu.com/s/1KD035L8UbrQrnd8hRNiB6A?pwd=fhyt). And then put the 'test' folder into the 'data/sritm-4k' folder.

Since the size of the SRITM-4K training set is too large, it will be uploaded soon.

# Test
You can run the following code to evaluate our FDAN:
```
# scale 2
python test.py\
    --config config/fdan_sritm4k_scale02.yaml\
    --pth model/fdan_sritm4k_scale02.pth\
    --scale 2\
    --test_hr data/sritm-4k/test/scale_02/hr/10bit\
    --test_lr data/sritm-4k/test/scale_02/lr/08bit\
    --evaluation_folder result\
    --exp_name FDAN_SRITM4K\
    --GT_norm 1023\
    --LQ_norm 255\
# scale 4
python test.py\
    --config config/fdan_sritm4k_scale04.yaml\
    --pth model/fdan_sritm4k_scale04.pth\
    --scale 4\
    --test_hr data/sritm-4k/test/scale_04/hr/10bit\
    --test_lr data/sritm-4k/test/scale_04/lr/08bit\
    --evaluation_folder result\
    --exp_name FDAN_SRITM4K\
    --GT_norm 1023\
    --LQ_norm 255\
# scale 8
python test.py\
    --config config/fdan_sritm4k_scale08.yaml\
    --pth model/fdan_sritm4k_scale08.pth\
    --scale 8\
    --test_hr data/sritm-4k/test/scale_08/hr/10bit\
    --test_lr data/sritm-4k/test/scale_08/lr/08bit\
    --evaluation_folder result\
    --exp_name FDAN_SRITM4K\
    --GT_norm 1023\
    --LQ_norm 255\
# scale 16
python test.py\
    --config config/fdan_sritm4k_scale16.yaml\
    --pth model/fdan_sritm4k_scale16.pth\
    --scale 16\
    --test_hr data/sritm-4k/test/scale_16/hr/10bit\
    --test_lr data/sritm-4k/test/scale_16/lr/08bit\
    --evaluation_folder result\
    --exp_name FDAN_SRITM4K\
    --GT_norm 1023\
    --LQ_norm 255\
```