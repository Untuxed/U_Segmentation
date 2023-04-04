import cv2

cap = cv2.VideoCapture('D:/OneDrive/OneDrive - University of Pittsburgh/Research/Projects/NormalData/UrethralShape/SymptomaticUrethralShape/urethraSlingStudy/data_fromHenry/Pre op/SS076_pre/SS076_20220302_150724_0000.avi')
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
