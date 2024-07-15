# Custom_Temporal Model
+ Language : Python3.7
+ Framework & Model :Pytorch, Mediapipe-pose, Custom_temporal_GCN
 ---
 ## Program Architecture
![image](https://github.com/user-attachments/assets/922f1f16-840e-475d-a199-9a1103bcc735)

---
## Model
 ### Dataset
![image](https://github.com/user-attachments/assets/6bd48008-6b05-4e5b-9729-f5c74b082483)

+ shape(프레임, 관절수, 좌표수)
---
### GCNLayer
![image](https://github.com/user-attachments/assets/0971acf4-b090-4f2f-951b-86c79281d1f5)

___
### GCNLayer
![image](https://github.com/user-attachments/assets/69048d60-1f81-460d-b77a-7045c4baba27)
___
## Model training before & after

![image](https://github.com/user-attachments/assets/b8eb0714-78fb-4251-8a52-24d3c8011468)
+ before training
     pooling: avarage pooling
    ◦ dropout: 마지막 GCNLayer 후 (rate0.2), 각 1dConvLayer 전 (rate0.4)
    ◦ activation function : relu (GCNLayer, 1DConvLayer)
    ◦ loss function : CrossEntropyLoss
    ◦ optimizer  : SGD
    ◦ learning rate : 0.01
    ◦ batchsize : 16
    ◦ ephoc : 30미만
___
![image](https://github.com/user-attachments/assets/40ee9d41-d6fb-460b-8082-1f8bab757457)
+ after trainig
     pooling: max pooling
    ◦ dropout: 마지막 GCNLayer 후 (rate0.5), 각 1dConvLayer 전 (rate0.4)
    ◦ activation function : relu (GCNLayer, 1DConvLayer)
    ◦ loss function : CrossEntropyLoss
    ◦ optimizer  : Adam
    ◦ learning rate : 0.0009
    ◦ batchsize : 16
    ◦ ephoc : 50
___
### Comparison with old model
+ old model
![image](https://github.com/user-attachments/assets/512cabec-2180-4be2-9502-0018a2baaf99)
+ new model
![image](https://github.com/user-attachments/assets/b56836d8-b6c7-4c38-99ed-7b361cb6ef35)

- inference 속도(total)
    기존 모델: cpu 166.79ms, gpu 166.89 
     재설계 모델: cpu 129.7ms, gpu 129.6ms
      -> 약 35ms 단축
- inference memory 사용률
    기존 모델: cpu 5.92%,  gpu 4.05%
     재설계 모델: cpu 1.63%, gpu 129.6ms
      -> 약 4% 단축
