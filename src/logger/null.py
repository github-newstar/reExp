from datetime import datetime


class NullWriter:
    """
    No-op experiment writer for sandbox/offline runs.
    Keeps the same interface as WandB/Comet writers.
    """

    def __init__(self, logger, project_config, **kwargs):
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode
        self.timer = datetime.now()

    def add_checkpoint(self, checkpoint_path, save_dir):
        return None

    def add_scalar(self, scalar_name, scalar):
        return None

    def add_scalars(self, scalars):
        return None

    def add_image(self, image_name, image):
        return None

    def add_audio(self, audio_name, audio, sample_rate=None):
        return None

    def add_text(self, text_name, text):
        return None

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        return None

    def add_table(self, table_name, table):
        return None

    def add_images(self, image_names, images):
        return None

    def add_pr_curve(self, curve_name, curve):
        return None

    def add_embedding(self, embedding_name, embedding):
        return None
