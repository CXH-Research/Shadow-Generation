# Generation

This is a shadow generation method. The shadow produced in this manner is of high quality. But it requires more resources.

Based on [GitHub - google/portrait-shadow-manipulation](https://github.com/google/portrait-shadow-manipulation)

Require tensorflow==1.15, opencv, numpy, pillow, tqdm and scipy module.

## Your folder structure should look like this:

```
Generation
 ┣ input(Place all your shadow-free images here)
 ┣ mask(Your mask images will be generated here)
 ┣ output(Your shadow images will be generated here)
 ┣ datasets.py
 ┣ example.png
 ┣ gen_mask.py
 ┣ gen_shadow.py
 ┣ utils.py
 ┗ README.md
```

## Command to run the program.

Use gen_shadow.py to generate shadow images
```python
python gen_shadow.py --min_val min_val --height image_height --width image_width  --num_shadow number of shadows
```
If you just want mask, you can use gen_mask.py
```python
python gen_mask.py --min_val min_val --height image_height --width image_width --num_mask number of masks
```

### Note
After finishing, it will generate a label.csv indicates the file path of input, mask and shadows respectively

Render per-pixel intensity variation mask within [min_val, 1.]

Only support jpg & png format images

num_shadow generate multiple shadow images at a time

## Example
![example](./example.png)