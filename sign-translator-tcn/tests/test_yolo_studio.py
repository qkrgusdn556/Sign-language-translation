import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import base64
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from yolo_studio import Studio


def picture(color):
    buffer = io.BytesIO()
    Image.new('RGB', (64, 48), color).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


class StudioTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.build')
        self.studio = Studio(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_and_delete(self):
        row = self.studio.add({'label': 'ㄱ', 'split': 'train', 'image': picture('red')})
        self.assertEqual(Studio(self.studio.root).rows[0]['label'], 'ㄱ')
        self.assertTrue((self.studio.root / 'images' / f'{row["id"]}.jpg').exists())
        self.studio.remove(row['id'])
        self.assertEqual(self.studio.status()['images'], [])

    def test_duplicate_across_splits_rejected(self):
        data = {'label': 'ㄱ', 'split': 'train', 'image': picture('red')}
        self.studio.add(data)
        with self.assertRaises(ValueError):
            self.studio.add({**data, 'split': 'val'})

    def test_bulk_remove_preserves_other_images_and_training_runs(self):
        rows = [self.studio.add({'label':'ㄱ','split':'train','image':picture(color)}) for color in ('red','green','blue')]
        model = self.studio.root / 'runs' / 'existing' / 'model.onnx'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'existing model')
        result = self.studio.remove_many([rows[0]['id'],rows[1]['id']])
        self.assertEqual(result['removed'],2)
        self.assertEqual(Studio(self.studio.root).rows,[rows[2]])
        self.assertEqual(model.read_bytes(),b'existing model')
        self.assertEqual(len(list((self.studio.root/'deleted').glob('*/*.jpg'))),2)
        self.assertTrue((self.studio.root/'images'/f'{rows[2]["id"]}.jpg').exists())

    def test_bulk_remove_invalid_selection_changes_nothing(self):
        row=self.studio.add({'label':'ㄱ','split':'train','image':picture('red')})
        for ids in ([],[row['id'],'../escape'],[row['id'],'0'*32],None):
            with self.assertRaises(ValueError):
                self.studio.remove_many(ids)
            self.assertEqual(Studio(self.studio.root).rows,[row])
            self.assertTrue((self.studio.root/'images'/f'{row["id"]}.jpg').exists())

    def test_training_requires_classes_and_validation(self):
        with self.assertRaises(ValueError):
            self.studio.train({'epochs': 1})
        for label in ['ㄱ', 'ㄴ']:
            for i in range(5):
                self.studio.add({'label': label, 'split': 'train', 'image': picture((i, 100 if label == 'ㄱ' else 200, 0))})
        with self.assertRaises(ValueError):
            self.studio.train({'epochs': 1})

    def test_invalid_inputs_and_paths(self):
        for value in [0, 201, True, '30']:
            with self.assertRaises(ValueError):
                self.studio.train({'epochs': value})
        with self.assertRaises(ValueError):
            self.studio.remove('../../images.json')
        with self.assertRaises(ValueError):
            self.studio.add({'label': '', 'split': 'train', 'image': picture('red')})
        with self.assertRaises(Exception):
            self.studio.add({'label': 'ㄱ', 'split': 'train', 'image': 'not an image'})

    def test_annotations_roundtrip_background_and_invalid_boxes(self):
        row = self.studio.add({'label': 'ㄱ', 'split': 'train', 'image': picture('red')})
        box = {'label': 'ㄴ', 'x': .2, 'y': .3, 'w': .4, 'h': .5}
        self.studio.annotate({'id': row['id'], 'boxes': [box]})
        stored = Studio(self.studio.root).rows[0]
        self.assertEqual(stored['label'], 'ㄱ')
        self.assertEqual(stored['boxes'], [box])
        for bad in [{**box,'x':-.1}, {**box,'w':1}, {**box,'h':float('nan')}, {**box,'label':''}]:
            with self.assertRaises(ValueError):
                self.studio.annotate({'id':row['id'],'boxes':[bad]})
        with self.assertRaises(ValueError):
            self.studio.annotate({'id':row['id'],'boxes':[]})
        self.assertEqual(self.studio.rows[0]['boxes'], [box])
        self.studio.annotate({'id':row['id'],'boxes':[],'background':True})
        self.assertTrue(self.studio.rows[0]['annotated'])
        self.assertEqual(self.studio.rows[0]['boxes'], [])


if __name__ == '__main__':
    unittest.main()
