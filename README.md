# 19496-Project

python main.py --fix_random --seed=10 --epoch=50 --dataset="muffle" --batch_size=128 --learning_rate_en=3e-4 --learning_rate_de=1e-4 --lamda=3e-2 --delta=1 --weight_decay=1e-5
```
    
```bash
python main.py --fix_random --seed=5 --epoch=40 --dataset="houston" --batch_size=256 --learning_rate_en=1e-4 --learning_rate_de=5e-4  --lamda=8e-2 --delta=0.5 --weight_decay=1e-5



### Generate Attribute Profile
The attribute profiles for LiDAR is applied based on the research work of [Dr. Liao](https://telin.ugent.be/~wliao/Partial_Reconstruction/). How to use the code can refer to this

```bash
num_scales = 2;

MPNeach=getmorphprofiles(ldr_image,'euclidean','square',num_scales);    

MPN=cat(3,MPNeach,ldr_image);


Reference:
Multimodal Hyperspectral Unmixing:
Insights From Attention Networks
Zhu Han , Student Member, IEEE , Danfeng Hong , Senior Member, IEEE,
Lianru Gao , Senior Member, IEEE, Jing Yao , Bing Zhang , Fellow, IEEE,
and Jocelyn Chanussot , Fellow, IEEE
