# -*- coding : utf-8-*-


class SavedModelDetails:

    def __init__(self, label_encoder, resize_width, resize_height):
        self.label_encoder = label_encoder
        self.resize_width = resize_width
        self.resize_height = resize_height

    def get_label_encoder(self):
        return self.label_encoder

    def get_resize_width(self):
        return self.resize_width

    def get_resize_height(self):
        return self.resize_height