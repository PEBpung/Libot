class Preprocessor(object):
    def __init__(self,
                 image_shape=(256, 256, 3),
                 heatmap_shape=(64, 64, 16),
                 is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)
        image = tf.io.decode_jpeg(features['image/encoded'])

        if self.is_train:
            random_margin = tf.random.uniform([1], 0.1, 0.3)[0]
            image, keypoint_x, keypoint_y = self.crop_roi(image, features, margin=random_margin)
            image = tf.image.resize(image, self.image_shape[0:2])
        else:
            image, keypoint_x, keypoint_y = self.crop_roi(image, features)
            image = tf.image.resize(image, self.image_shape[0:2])

        image = tf.cast(image, tf.float32) / 127.5 - 1
        heatmaps = self.make_heatmaps(features, keypoint_x, keypoint_y)

        # print (image.shape, heatmaps.shape, type(heatmaps))

        return image, heatmaps

    def crop_roi(self, image, features, margin=0.2):
            img_shape = tf.shape(image)
            img_height = img_shape[0]
            img_width = img_shape[1]
            img_depth = img_shape[2]

            keypoint_x = tf.cast(tf.sparse.to_dense(features['image/object/parts/x']), dtype=tf.int32)
            keypoint_y = tf.cast(tf.sparse.to_dense(features['image/object/parts/y']), dtype=tf.int32)
            center_x = features['image/object/center/x']
            center_y = features['image/object/center/y']
            body_height = features['image/object/scale'] * 200.0

            # keypoint 중 유효한값(visible = 1) 만 사용합니다.
            masked_keypoint_x = tf.boolean_mask(keypoint_x, keypoint_x > 0)
            masked_keypoint_y = tf.boolean_mask(keypoint_y, keypoint_y > 0)

            # min, max 값을 찾습니다.
            keypoint_xmin = tf.reduce_min(masked_keypoint_x)
            keypoint_xmax = tf.reduce_max(masked_keypoint_x)
            keypoint_ymin = tf.reduce_min(masked_keypoint_y)
            keypoint_ymax = tf.reduce_max(masked_keypoint_y)

            # 높이 값을 이용해서 x, y 위치를 재조정 합니다. 박스를 정사각형으로 사용하기 위해 아래와 같이 사용합니다.
            xmin = keypoint_xmin - tf.cast(body_height * margin, dtype=tf.int32)
            xmax = keypoint_xmax + tf.cast(body_height * margin, dtype=tf.int32)
            ymin = keypoint_ymin - tf.cast(body_height * margin, dtype=tf.int32)
            ymax = keypoint_ymax + tf.cast(body_height * margin, dtype=tf.int32)

            # 이미지 크기를 벗어나는 점을 재조정 해줍니다.
            effective_xmin = xmin if xmin > 0 else 0
            effective_ymin = ymin if ymin > 0 else 0
            effective_xmax = xmax if xmax < img_width else img_width
            effective_ymax = ymax if ymax < img_height else img_height
            effective_height = effective_ymax - effective_ymin
            effective_width = effective_xmax - effective_xmin

            image = image[effective_ymin:effective_ymax, effective_xmin:effective_xmax, :]
            new_shape = tf.shape(image)
            new_height = new_shape[0]
            new_width = new_shape[1]

            # shift all keypoints based on the crop area
            effective_keypoint_x = (keypoint_x - effective_xmin) / new_width
            effective_keypoint_y = (keypoint_y - effective_ymin) / new_height

            return image, effective_keypoint_x, effective_keypoint_y
