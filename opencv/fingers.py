import cv2
import numpy as np

class HandTracker:
    """
    Класс для обнаружения и трекинга пальцев руки в видеопотоке
    """
    def __init__(self):
        self.skin_lower = np.array([0, 30, 60], dtype=np.uint8)
        self.skin_upper = np.array([20, 150, 255], dtype=np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)
        self.epsilon_factor = 0.02
        
    def preprocess_frame(self, frame):
        """
        Предобработка кадра: размытие, конверсия в HSV, морфологические операции
        """
        blurred = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def find_hand_contour(self, mask):
        """
        Поиск контура руки как наибольшего связного компонента на маске
        """
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)        
        if not contours:
            return None
        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) < 5000:
            return None
            
        return hand_contour
    
    def find_finger_tips(self, contour, hull_indices):
        """
        Обнаружение кончиков пальцев на основе анализа дефектов выпуклости
        """
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return []
        finger_tips = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / np.pi
            if angle < 90:
                finger_tips.append(start)
                finger_tips.append(end)
        finger_tips = list(set(finger_tips))
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            finger_tips = [pt for pt in finger_tips if pt[1] < cy - 20]
        
        return finger_tips
    
    def count_fingers(self, contour):
        """
        Подсчет количества поднятых пальцев на основе анализа выпуклостей
        """
        hull = cv2.convexHull(contour, returnPoints=False)
        finger_tips = self.find_finger_tips(contour, hull)
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return 0            
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        distances = []
        for point in contour:
            px, py = point[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            distances.append(dist)
        
        if not distances:
            return 0
            
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        peaks = 0
        threshold = avg_dist + 0.5 * std_dist
        
        for point in contour:
            px, py = point[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            
            if dist > threshold:
                if py < cy:
                    peaks += 1
        return min(peaks, 5)
    
    def draw_hand_info(self, frame, contour, finger_count):
        """
        Визуализация результатов на кадре
        """
        if contour is not None:
            # Отрисовка контура руки зеленым цветом
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Отрисовка выпуклой оболочки синим цветом
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
            
            # Вычисление ограничивающего прямоугольника
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Отображение количества пальцев
            text = f"Fingers: {finger_count}"
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            
            # Поиск и отображение кончиков пальцев
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            finger_tips = self.find_finger_tips(contour, hull_indices)
            
            for tip in finger_tips:
                cv2.circle(frame, tip, 8, (0, 0, 255), -1)  # Красные круги на кончиках
                cv2.circle(frame, tip, 10, (255, 255, 255), 2)  # Белая обводка
        
        return frame
    
    def run(self):
        """
        Основной цикл обработки видеопотока
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Ошибка инициализации камеры")
            return
        cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL) 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            mask = self.preprocess_frame(frame)
            hand_contour = self.find_hand_contour(mask)
            finger_count = 0
            if hand_contour is not None:
                finger_count = self.count_fingers(hand_contour)
                display_frame = self.draw_hand_info(
                    display_frame, hand_contour, finger_count
                )
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((display_frame, mask_bgr))
            cv2.imshow('Hand Tracking', combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
