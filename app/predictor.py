from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Concatenate, 
                                     TimeDistributed, GlobalAveragePooling1D, Dropout)
from tensorflow.keras.models import Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Глобальные параметры, используемые при обучении (убедитесь, что они совпадают)
ORIGINAL_SIZE = (1024, 1024)   # исходное разрешение
PATCH_SIZE = (224, 224)        # размер патча
GRID_SIZE = (1, 1)             # равномерная сетка (здесь для примера 1x1; замените на нужное значение, например, (4,4))
NUM_PATCHES = GRID_SIZE[0] * GRID_SIZE[1]  # ожидаемое число патчей

def ordinal_mae(y_true, y_pred):
    """
    Кастомная метрика для вычисления MAE для ordinal задач.
    y_pred имеет форму (batch, num_classes) – распределение вероятностей.
    y_true имеет форму (batch,) или (batch,1) – скалярные истинные метки.
    Вычисляем argmax по y_pred и затем среднюю абсолютную ошибку.
    """
    # Приводим y_true к форме (batch,) и преобразуем в int32
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    # Получаем предсказанные метки как argmax
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    # Вычисляем абсолютную ошибку
    error = tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred_labels, tf.float32))
    return tf.reduce_mean(error)
    
# ---------------------------
# 4. Взвешенная бинарная кроссэнтропия для задачи defect
# ---------------------------
def weighted_binary_crossentropy(pos_weight):
    """
    pos_weight – вес для положительного класса.
    Если положительный класс встречается реже, pos_weight > 1.
    """
    def loss(y_true, y_pred):
        # Стандартная бинарная кроссэнтропия
        bce = K.binary_crossentropy(y_true, y_pred)
        # Вес для каждого примера: если y_true == 1, вес = pos_weight, иначе 1.
        weight_vector = y_true * pos_weight + (1.0 - y_true)
        return K.mean(bce * weight_vector)
    return loss

# ---------------------------
# 2. Функция потерь, основанная на Earth Mover’s Distance (EMD) для ordinal задач
# (здесь мы суммируем разницу между накопительными распределениями)
# ---------------------------
def ordinal_emd_loss(num_classes):
    """
    num_classes – число классов в данной задаче.
    Эта функция предполагает, что y_true имеет форму (batch, 1) (скалярные метки),
    а y_pred – распределение вероятностей по классам (batch, num_classes).
    """
    def loss(y_true, y_pred):
        # Преобразуем y_true в one-hot (batch, num_classes)
        y_true_onehot = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=num_classes)
        # Вычисляем накопительные суммы (CDF)
        cdf_true = tf.math.cumsum(y_true_onehot, axis=1)
        cdf_pred = tf.math.cumsum(y_pred, axis=1)
        # EMD – средняя абсолютная разность между CDF
        return tf.reduce_mean(tf.abs(cdf_true - cdf_pred))
    return loss

### MODEL

def create_patch_feature_extractor(patch_size):
    """
    Создает модель для извлечения признаков из одного патча.
    Использует ResNet50 (без верхней части) + GlobalAveragePooling2D и Dense.
    """
    # Явно приводим размеры к int, на случай если patch_size передается не в виде целых чисел.
    patch_h = int(patch_size[0])
    patch_w = int(patch_size[1])
    
    patch_input = Input(shape=(patch_h, patch_w, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(patch_h, patch_w, 3))
    base_model.trainable = False
    x = base_model(patch_input)
    x = GlobalAveragePooling2D()(x)
    out = Dense(128, activation='relu')(x)
    return Model(inputs=patch_input, outputs=out, name='patch_feature_extractor')

def create_multi_task_model(patch_size, grid_size):
    """
    Строит модель, которая получает на вход набор патчей для реверса и аверса,
    извлекает общие признаки, агрегирует их и имеет 5 выходов:
      - defect (бинарная)
      - preservation_defect (8 классов)
      - preservation_no_defect (28 классов)
      - prefix (3 класса)
      - postfix (4 класса)
    """
    
    
    # Входы: для каждого изображения набор патчей
    num_patches = grid_size[0] * grid_size[1]   # Это уже int, если grid_size – кортеж чисел
    
    input_revers = Input(shape=(num_patches, patch_size[0], patch_size[1], 3), name='revers_input')
    input_avers  = Input(shape=(num_patches, patch_size[0], patch_size[1], 3), name='avers_input')
    
    # Создаем общий feature extractor для патчей
    patch_extractor = create_patch_feature_extractor(patch_size)
    
    # Применяем его к каждому патчу с помощью TimeDistributed
    revers_features = TimeDistributed(patch_extractor, name='td_revers')(input_revers)  # (batch, patches, 128)
    avers_features  = TimeDistributed(patch_extractor, name='td_avers')(input_avers)    # (batch, patches, 128)
    
    # Агрегируем признаки по патчам (усреднение)
    revers_agg = GlobalAveragePooling1D(name='gap_revers')(revers_features)  # (batch, 128)
    avers_agg  = GlobalAveragePooling1D(name='gap_avers')(avers_features)    # (batch, 128)
    
    # Объединяем признаки двух сторон
    combined = Concatenate(name='combined')([revers_agg, avers_agg])  # (batch, 256)
    shared_fc = Dense(256, activation='relu', name='shared_fc1')(combined)
    shared_fc = Dropout(0.5, name='dropout_shared')(shared_fc)
    
    #############################################
    # Голова 1: Определение наличия дефектов (бинарная классификация)
    defect_out = Dense(1, activation='sigmoid', name='defect')(shared_fc)
    
    #############################################
    # Голова 2: Сохранность для монет с дефектами (8 классов)
    pres_def_fc = Dense(128, activation='relu', name='pres_def_fc')(shared_fc)
    preservation_defect_out = Dense(8, activation='softmax', name='preservation_defect')(pres_def_fc)
    
    #############################################
    # Голова 3: Сохранность для монет без дефектов (28 классов)
    pres_no_def_fc = Dense(128, activation='relu', name='pres_no_def_fc')(shared_fc)
    preservation_no_defect_out = Dense(28, activation='softmax', name='preservation_no_defect')(pres_no_def_fc)
    
    #############################################
    # Голова 4: Префикс (3 класса: нет, PL, PF)
    prefix_fc = Dense(64, activation='relu', name='prefix_fc')(shared_fc)
    prefix_out = Dense(3, activation='softmax', name='prefix')(prefix_fc)
    
    #############################################
    # Голова 5: Постфикс (4 класса: нет, BN, RB, RD)
    postfix_fc = Dense(64, activation='relu', name='postfix_fc')(shared_fc)
    postfix_out = Dense(4, activation='softmax', name='postfix')(postfix_fc)
    
    model = Model(
        inputs=[input_revers, input_avers],
        outputs={
            "defect": defect_out,
            "preservation_defect": preservation_defect_out,
            "preservation_no_defect": preservation_no_defect_out,
            "prefix": prefix_out,
            "postfix": postfix_out
        }
    )



    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "defect": weighted_binary_crossentropy(pos_weight=5.0),  # задайте pos_weight в зависимости от дисбаланса
            "preservation_defect": ordinal_emd_loss(8),               # 8 классов для монет с дефектами
            "preservation_no_defect": ordinal_emd_loss(28),             # 28 классов для монет без дефектов
            "prefix": "sparse_categorical_crossentropy",              # 3 класса: нет, PL, PF
            "postfix": "sparse_categorical_crossentropy"              # 4 класса: нет, BN, RB, RD
        },
        loss_weights={
            "defect": 1.0,
            "preservation_defect": 1.0,
            "preservation_no_defect": 1.0,
            "prefix": 0.5,    # если эти задачи менее важны, можно снизить их вклад
            "postfix": 0.5
        },
        metrics={
            "defect": tf.keras.metrics.AUC(name="auc"),
            "preservation_defect": [ordinal_mae],
            "preservation_no_defect": [ordinal_mae],
            "prefix": "accuracy",
            "postfix": "accuracy"
        }
    )
    return model  

model = create_multi_task_model(PATCH_SIZE, GRID_SIZE)
model.load_weights("coin_condition_1.keras")


def apply_circular_mask(img_array):
    """
    Принимает изображение (H, W, 3) и обнуляет все пиксели вне круга, вписанного в изображение.
    """
    H, W, _ = img_array.shape
    center_y, center_x = H / 2, W / 2
    radius = min(H, W) / 2
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist_from_center <= radius
    mask = np.expand_dims(mask, axis=-1).astype(img_array.dtype)
    return img_array * mask

def extract_grid_patches(img_array, patch_size, grid_size):
    """
    Разбивает изображение (уже замаскированное) на равномерную сетку патчей.
    Возвращает массив патчей формы (rows*cols, patch_h, patch_w, 3).
    """
    H, W, _ = img_array.shape
    rows, cols = grid_size
    patch_h, patch_w = patch_size
    row_step = (H - patch_h) // (rows - 1) if rows > 1 else 0
    col_step = (W - patch_w) // (cols - 1) if cols > 1 else 0

    patches = []
    for i in range(rows):
        for j in range(cols):
            y = i * row_step
            x = j * col_step
            patch = img_array[y:y+patch_h, x:x+patch_w, :]
            patches.append(patch)
    patches = np.array(patches)
    # Если извлечено меньше патчей, чем ожидалось, заполним недостающие копиями первого патча
    if patches.shape[0] < NUM_PATCHES:
        missing = NUM_PATCHES - patches.shape[0]
        if patches.shape[0] > 0:
            extra = np.tile(patches[0:1], (missing, 1, 1, 1))
        else:
            extra = np.zeros((missing, patch_size[0], patch_size[1], 3), dtype=img_array.dtype)
        patches = np.concatenate([patches, extra], axis=0)
    # Если извлечено больше патчей, обрежем
    if patches.shape[0] > NUM_PATCHES:
        patches = patches[:NUM_PATCHES]
    return patches

def preprocess_image_for_prediction(image_path):
    """
    Загружает изображение по image_path, изменяет его размер до ORIGINAL_SIZE,
    применяет круговую маску, равномерно разбивает изображение на патчи,
    и применяет предобработку для ResNet50.
    Возвращает массив патчей формы (NUM_PATCHES, patch_h, patch_w, 3).
    """
    img = load_img(image_path, target_size=ORIGINAL_SIZE)
    img_array = img_to_array(img)
    img_array = apply_circular_mask(img_array)
    patches = extract_grid_patches(img_array, PATCH_SIZE, GRID_SIZE)
    patches = tf.keras.applications.resnet50.preprocess_input(patches)
    return patches

# Обратные маппинги для интерпретации предсказаний
defective_mapping_inv = {
    0: "PF det.",
    1: "UNC det.",
    2: "AU det.",
    3: "XF det.",
    4: "VF det.",
    5: "F det.",
    6: "VG det.",
    7: "G det."
}
no_def_allowed = [4,6,8,10,12,15,20,25,30,35,40,45,50,53,55,58,60,61,62,63,64,65,66,67,68,69,70]
no_def_mapping_inv = {i: str(val) for i, val in enumerate(no_def_allowed)}
prefix_mapping_inv = {0: "", 1: "PL", 2: "PF"}
postfix_mapping_inv = {0: "", 1: "BN", 2: "RB", 3: "RD"}

# Функция, которая принимает пути к изображениям и возвращает оценку монеты
def predict_coin_condition(avers_path, revers_path):
    """
    Принимает пути к изображению аверса и реверса, выполняет предобработку,
    получает предсказание от модели и возвращает словарь с результатами.
    """
    # Получаем патчи для каждого изображения
    patches_avers = preprocess_image_for_prediction(avers_path)  # (NUM_PATCHES, 224, 224, 3)
    patches_revers = preprocess_image_for_prediction(revers_path)
    
    # Добавляем размерность батча: итоговая форма (1, NUM_PATCHES, 224, 224, 3)
    X_avers = np.expand_dims(patches_avers, axis=0)
    X_revers = np.expand_dims(patches_revers, axis=0)
    
    # Порядок входов: проверьте, какой порядок использовался при обучении.
    # Здесь предполагается, что модель ожидает [revers, avers]
    preds = model.predict([X_revers, X_avers], verbose=0)
    
    # Распаковываем предсказания (адаптируйте под вашу модель)
    pred_defect = preds["defect"][0][0]
    pred_pres_def = preds["preservation_defect"][0]
    pred_pres_no_def = preds["preservation_no_defect"][0]
    pred_prefix = preds["prefix"][0]
    pred_postfix = preds["postfix"][0]
    
    has_defect = pred_defect >= 0.5
    if has_defect:
        cls_idx = np.argmax(pred_pres_def)
        condition_str = defective_mapping_inv.get(cls_idx, "Неопределено")
    else:
        cls_idx = np.argmax(pred_pres_no_def)
        numeric_str = no_def_mapping_inv.get(cls_idx, "")
        if numeric_str and int(numeric_str) > 53:
            pref_idx = np.argmax(pred_prefix)
            prefix_str = prefix_mapping_inv.get(pref_idx, "")
        else:
            prefix_str = ""
        post_idx = np.argmax(pred_postfix)
        postfix_str = postfix_mapping_inv.get(post_idx, "")
        condition_str = f"{prefix_str}{numeric_str}{postfix_str}"
    
    result = {
        "defect_probability": float(pred_defect),
        "condition": condition_str
    }
    return result
