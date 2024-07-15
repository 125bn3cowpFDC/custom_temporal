import mediapipe as mp
import cv2
import numpy as np

import torch
import torch.nn as nn

import ChoomModel
import processcopy
def point_return(landmark):
    a = []

    a.append(round(landmark[0].x, 3))
    a.append(round(landmark[0].y, 3))

    a.append(round(landmark[2].x, 3))
    a.append(round(landmark[2].y, 3))

    a.append(round(landmark[5].x, 3))
    a.append(round(landmark[5].y, 3))

    a.append(round(landmark[7].x, 3))
    a.append(round(landmark[7].y, 3))

    a.append(round(landmark[8].x, 3))
    a.append(round(landmark[8].y, 3))

    a.append(round(landmark[11].x, 3))
    a.append(round(landmark[11].y, 3))

    a.append(round(landmark[12].x, 3))
    a.append(round(landmark[12].y, 3))

    a.append(round(landmark[13].x, 3))
    a.append(round(landmark[13].y, 3))

    a.append(round(landmark[14].x, 3))
    a.append(round(landmark[14].y, 3))

    a.append(round(landmark[15].x, 3))
    a.append(round(landmark[15].y, 3))

    a.append(round(landmark[16].x, 3))
    a.append(round(landmark[16].y, 3))

    a.append(round(landmark[23].x, 3))
    a.append(round(landmark[23].y, 3))

    a.append(round(landmark[24].x, 3))
    a.append(round(landmark[24].y, 3))

    a.append(round(landmark[25].x, 3))
    a.append(round(landmark[25].y, 3))

    a.append(round(landmark[26].x, 3))
    a.append(round(landmark[26].y, 3))

    a.append(round(landmark[27].x, 3))
    a.append(round(landmark[27].y, 3))

    a.append(round(landmark[28].x, 3))
    a.append(round(landmark[28].y, 3))

    a.append(round((landmark[11].x + landmark[12].x)/2, 3))
    a.append(round((landmark[11].y + landmark[12].y)/2, 3))

    return a
def main():
    labeling = {}
    cnt = 0
    show_cnt = 0
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = ChoomModel.Mymodel()
    model.load_state_dict(torch.load('./te50st10lr0.0009b16bcafterbe.pt'))
    model = model.to(device)
    model.eval()

    data_numpy = np.zeros((100, 18, 2))
    admatrix = processcopy.get_admatrix(device).expand(1,100,18,18).contiguous()
    label_path = './8class_labels.txt'

    with open(label_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            strings = line.split(' ')
            labeling[int(strings[0])] = strings[1][:-1]

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    start_frame = 0
    videopath = "./mix_clear.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    cap = cv2.VideoCapture(videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        #outputs = model(data_tensor)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("afaafa")
                print('CANT OPEN VIDEO')
                break

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # pose detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Get landmarks
                print('*** NOW_FRAME: ', cnt)
                landmarks = results.pose_landmarks.landmark
                landmark_list = point_return(landmarks) # type:list
                #print(landmark_list)

                if cnt < 100:
                    data_numpy[cnt,:,0] = landmark_list[0::2]
                    data_numpy[cnt,:,1] = landmark_list[1::2]
                elif cnt >= 100:
                    for i in range(99):
                        data_numpy[i, :, 0] = data_numpy[i+1, :, 0]
                        data_numpy[i, :, 1] = data_numpy[i+1, :, 1]
                    data_numpy[99, :, 0] = landmark_list[0::2]
                    data_numpy[99, :, 1] = landmark_list[1::2] 
                    data_numpy = data_numpy
                    if cnt % 10 == 0:
                        print(data_numpy.shape)
                    
                        
                        data_tensor = torch.from_numpy(data_numpy).float().to(device)
                        data_tensor = data_tensor.view(1,100,18,2).contiguous()
                        
                
                        print(data_tensor.shape)
                        with torch.no_grad():
                            try:
                                outputs = model(data_tensor,admatrix)
                            except Exception as e:
                                print(f"Exception occurred: {e}")
                            
                        predicted = torch.argmax(outputs,1)
                        pred = predicted.item()
                        softmax_out = torch.nn.functional.softmax(outputs, dim=1)*100
                        #softmax_out = softmax_out.to('cpu').numpy()
                        #np.set_printoptions(precision=2, suppress=True)
                        show_cnt += 1
                        
                    
                    label_name = labeling[int(pred)]
                    this_out = round(float(softmax_out[0, int(pred)]),2)
                    cv2.putText(image, label_name +':  ' + str(this_out), (800, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 2, cv2.LINE_AA)
                    
                    breathing = float(softmax_out[0, 0])
                    walking_sawi = float(softmax_out[0, 1])
                    jump_sawi = float(softmax_out[0, 2])
                    turn_sawi = float(softmax_out[0, 3])
                    normp_sawi = float(softmax_out[0, 4])
                    normp_sawi2 = float(softmax_out[0, 5])
                    everyone_sawi = float(softmax_out[0, 6])
                    wind_sawi = float(softmax_out[0, 7])

                    breathing = int(breathing)
                    walking_sawi = int(walking_sawi)
                    jump_sawi = round(jump_sawi,2)
                    turn_sawi = round(turn_sawi,2)
                    normp_sawi = round(normp_sawi,2)
                    normp_sawi2 = round(normp_sawi2,2)
                    everyone_sawi = round(everyone_sawi,2)
                    wind_sawi = round(wind_sawi,2)

                    cv2.putText(image, 'breathing: ' + str(breathing), (800, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'walking_sawi: ' + str(walking_sawi), (800, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'jump_sawi: ' + str(jump_sawi), (800, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'turn_sawi: ' + str(turn_sawi), (800, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi: ' + str(normp_sawi), (800, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi2: ' + str(normp_sawi2), (800, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'everyone_sawi: ' + str(everyone_sawi), (800, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'wind_sawi: ' + str(wind_sawi), (800, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    
                cv2.putText(image, 'predict conunt: ' + str(show_cnt), (800, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'pos frame: ' + str(cnt), (800, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)


            
                
            cnt += 1
            cv2.imshow('Mediapipe Feeder', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("afaaf")
                break 
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
