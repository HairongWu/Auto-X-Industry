import os
from PIL import Image
from sklearn.model_selection import train_test_split
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

from .pipeline import *
from .ultralytics import YOLOv10

tform= transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class LSPRPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

        self.model_path = os.environ.get("MODEL_POOL_DIR")
        self.model = YOLOv10(os.path.join(self.model_path, "yolov10x.pt"))

        self.save_dir=os.path.join(os.environ.get("MODEL_POOL_DIR"), "lspr")
        self.conf_threshold = 0.25

        args = AttrDict()
        embedding = models.resnet101(pretrained=True)
        return_nodes = {
            'avgpool': 'embedding'
        }
        self.embedding = create_feature_extractor(embedding, return_nodes=return_nodes)

        self.embedding.eval()
        self.embedding.to(self.device)

        vector_store_path = os.path.join(self.model_path,'autox_lspr')
        if os.path.exists(vector_store_path):
            self.vector_store = VectorStore(
                path = vector_store_path
            )
        else:
            self.vector_store = VectorStore(
                path = vector_store_path,
                tensor_params = [{'name': 'embedding', 'htype': 'embedding'}, 
                                {'name': 'description', 'htype': 'tag'}],
            )

    def predict(self, image_paths):
        # Fix me: Change to batch inferences
        all_boxes = []
        all_labels = []
        all_logits = []
        all_lengths = []

        for img in image_paths:
            boxes_filt = self.model.predict(source=img, save=False, conf=self.conf_threshold)[0]
            boxes = boxes_filt.boxes
            labels = boxes_filt.names
            cls = boxes.cls.cpu().numpy()
            cls = [labels[c] for c in cls]

            cls = self.get_from_vdb(img, boxes, cls)

            all_boxes.append(boxes.xyxy.cpu().numpy())
            all_lengths.append(boxes.orig_shape)
            all_logits.append(boxes.conf.cpu().numpy())
            all_labels.append(cls)
            
        return all_boxes, all_labels, all_logits, all_lengths

    def train(self, annotations, names):
        self.save_to_vdb(annotations, names)

        train, val = train_test_split(annotations, test_size=0.4)
        val, test = train_test_split(val, test_size=0.5)

        data = {"train":train,"val":val, "nc":len(names), "names":names}
        self.model.train(data=data, epochs=500, batch=8, imgsz=320)
        data = {"test":test}
        self.model.val(data=data, batch=256)

    def finetune(self, annotations):
        pass

    def get_from_vdb(self, image, boxes, cls):
        return cls

    def embedding_function(self, images, transform = tform, batch_size = 4):

        #Proceess the embeddings in batches, but return everything as a single list
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([transform(item) for item in images[i:i+batch_size]])
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings+= self.embedding(batch)['embedding'][:,:,0,0].cpu().numpy().tolist()

        return embeddings
    def save_to_vdb(self, annotations, names):
        images = []
        des = []
        for annos in annotations:
            des.append(names[annos['cls'][0][0]])
            boxes = annos['bboxes'][0]
            im_file = Image.open(annos['im_file'])
            images.append(im_file.crop(boxes))


        self.vector_store.add(
                 description = des,
                 embedding_function = self.embedding_function, 
                 embedding_data = images)