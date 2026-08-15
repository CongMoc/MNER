import torch
import logging
import os
import random
logger = logging.getLogger(__name__)
from torchvision import transforms
from PIL import Image
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
URL_PREFIX = 'http'
class SBInputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,guid,text_a,text_b,img_id,label=None,auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a=text_a
        self.text_b=text_b
        self.img_id = img_id
        self.label = label
        # Please note that the auxlabel is not used in SB
        # it is just kept in order not to modify the original code
        self.auxlabel = auxlabel

class SBInputFeatures(object):
    """A single set of features of data"""

    def __init__(self,input_ids,input_mask,added_input_mask,segment_ids,img_feat,img_ti_feat,label_id,auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.img_ti_feat= img_ti_feat
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id

def sbreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    print("prepare data for ",filename)
    f = open(filename,encoding='utf8')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    imgid = ''
    a = 0
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1] + '.jpg'
            continue

        if line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        
        if splits[0] == '' or splits[0].isspace() or splits[0] in SPECIAL_TOKENS or splits[0].startswith(URL_PREFIX):
            splits[0] = "<unk>"
        elif ' ' in splits[0]:
            # a token must not contain whitespace: text_a is later reconstructed via
            # ' '.join(sentence) and re-split on ' ', so an embedded space here would
            # desync it from the per-token label list
            splits[0] = '_'.join(splits[0].split())

        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) > 0:
        data.append((sentence, label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: " + str(len(data)))
    print("The number of images: " + str(len(imgs)))
    return data, imgs, auxlabels

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_sbtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return sbreadfile(input_file)

class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "dev.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_sbtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")

    def get_labels(self):
        return os.getenv("LABELS", "").split(",")
        # vlsp2021
        # return ["O","I-PRODUCT-AWARD","B-MISCELLANEOUS","B-QUANTITY-NUM","B-ORGANIZATION-SPORTS","B-DATETIME","I-ADDRESS","I-PERSON","I-EVENT-SPORT","B-ADDRESS","B-EVENT-NATURAL","I-LOCATION-GPE","B-EVENT-GAMESHOW","B-DATETIME-TIMERANGE","I-QUANTITY-NUM","I-QUANTITY-AGE","B-EVENT-CUL","I-QUANTITY-TEM","I-PRODUCT-LEGAL","I-LOCATION-STRUC","I-ORGANIZATION","B-PHONENUMBER","B-IP","B-QUANTITY-AGE","I-DATETIME-TIME","I-DATETIME","B-ORGANIZATION-MED","B-DATETIME-SET","I-EVENT-CUL","B-QUANTITY-DIM","I-QUANTITY-DIM","B-EVENT","B-DATETIME-DATERANGE","I-EVENT-GAMESHOW","B-PRODUCT-AWARD","B-LOCATION-STRUC","B-LOCATION","B-PRODUCT","I-MISCELLANEOUS","B-SKILL","I-QUANTITY-ORD","I-ORGANIZATION-STOCK","I-LOCATION-GEO","B-PERSON","B-PRODUCT-COM","B-PRODUCT-LEGAL","I-LOCATION","B-QUANTITY-TEM","I-PRODUCT","B-QUANTITY-CUR","I-QUANTITY-CUR","B-LOCATION-GPE","I-PHONENUMBER","I-ORGANIZATION-MED","I-EVENT-NATURAL","I-EMAIL","B-ORGANIZATION","B-URL","I-DATETIME-TIMERANGE","I-QUANTITY","I-IP","B-EVENT-SPORT","B-PERSONTYPE","B-QUANTITY-PER","I-QUANTITY-PER","I-PRODUCT-COM","I-DATETIME-DURATION","B-LOCATION-GPE-GEO","B-QUANTITY-ORD","I-EVENT","B-DATETIME-TIME","B-QUANTITY","I-DATETIME-SET","I-LOCATION-GPE-GEO","B-ORGANIZATION-STOCK","I-ORGANIZATION-SPORTS","I-SKILL","I-URL","B-DATETIME-DURATION","I-DATETIME-DATE","I-PERSONTYPE","B-DATETIME-DATE","I-DATETIME-DATERANGE","B-LOCATION-GEO","B-EMAIL","X","<s>", "</s>"]
        
        # vlsp2016
        return ["O","B-ORG","B-MISC","I-PER","I-ORG","B-LOC","I-MISC","I-LOC","B-PER","X","<s>","</s>"]

        # vlsp2018
        # return ["O","I-ORGANIZATION","B-ORGANIZATION","I-LOCATION","B-MISCELLANEOUS","I-PERSON","B-PERSON","I-MISCELLANEOUS","B-LOCATION","X","<s>","</s>"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "<s>", "</s>"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['<s>']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['</s>']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            examples.append(
                SBInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
        return examples


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def convert_mm_examples_to_features(examples, label_list, auxlabel_list,
 max_seq_length, tokenizer, crop_size, path_img, num_image_tokens=49,
 image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
 random_image_source=False, random_seed=37, blank_image_source=False,
 generated_image_source=False):
    # num_image_tokens/image_mean/image_std default to the ResNet-152 setup
    # (7x7=49 spatial patches, ImageNet stats); pass ViT-specific values
    # (e.g. 196 patches, mean/std=0.5) when the visual backbone is a ViT.
    # random_image_source: ablation mode ("Random" image source) -- each
    # example gets a fixed, randomly-chosen real image from path_img instead
    # of its correct IMGID-matched image. The mapping is generated once here
    # (deterministic given random_seed) and then baked into the cached
    # features, so it stays fixed across the whole run.
    # blank_image_source: ablation mode ("Blank" image source) -- every
    # example uses the same neutral background.jpg placeholder instead of
    # its correct image, so the model gets no per-example visual signal.
    # generated_image_source: ablation mode ("Generated" image source) -- each
    # example uses a pre-generated (e.g. text-to-image) image stored at
    # path_img/{example.guid}.jpg instead of its correct crawled image.

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0
    ti_crop_size=32

    if random_image_source:
        image_pool = sorted(f for f in os.listdir(path_img) if f != 'background.jpg')
        rng = random.Random(random_seed)

    transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)])

    transform_for_ti = transforms.Compose([
            transforms.Resize([ti_crop_size, ti_crop_size]),  # 调整图片到指定的大小
            transforms.ToTensor(),
            transforms.Normalize((0.48, 0.498, 0.531),
                                (0.214, 0.207, 0.207))])

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("<s>")
        segment_ids.append(0)
        label_ids.append(label_map["<s>"])
        auxlabel_ids.append(auxlabel_map["<s>"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("</s>")
        segment_ids.append(0)
        label_ids.append(label_map["</s>"])
        auxlabel_ids.append(auxlabel_map["</s>"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + num_image_tokens)  # extra tokens for encoding regional image representations

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        if blank_image_source:
            image_name = 'background.jpg'
        elif random_image_source:
            image_name = rng.choice(image_pool)
        elif generated_image_source:
            image_name = f"{example.guid}.jpg"
        else:
            image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            if 'NaN' not in image_path:
                print(image_path)
        try:
            image = image_process(image_path, transform)
            image_ti_feat = image_process(image_path, transform_for_ti)
        except:
            count += 1
            image_path_fail = os.path.join(path_img, 'background.jpg')
            image = image_process(image_path_fail, transform)
            image_ti_feat = image_process(image_path_fail, transform_for_ti)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            # logger.info(f"img_feat: {image.shape}")
            # logger.info(f"img_ti_feat: {image_ti_feat.shape}")

            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            SBInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                            segment_ids=segment_ids, img_feat=image, img_ti_feat=image_ti_feat, label_id=label_ids, auxlabel_id=auxlabel_ids))

    print('the number of problematic samples: ' + str(count))
    return features


if __name__ == "__main__":
    processor = MNERProcessor()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    data_dir = r'/home/vms/bags/vlsp_all/origin+image/VLSP2018'
    eval_examples = processor.get_test_examples(data_dir)
    print(eval_examples[0].img_id)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, 256, tokenizer, 224, data_dir+'/ner_image')
    
    print(len(eval_features))