# Generation

This is a shadow generation method. The shadow produced in this manner is of high quality. But it requires more resources.

Based on [GitHub - google/portrait-shadow-manipulation](https://github.com/google/portrait-shadow-manipulation)

Require tensorflow==1.15, opencv, numpy and scipy module.

## Your folder structure should look like this:

```
Generation
 ┣ input(Place all your shadow-free images here)
 ┣ mask(Place all your shadow mask images here)
 ┣ output(Your shadow images will be generated here)
 ┣ gen_shadow.py
 ┣ datasets.py
 ┣ utils.py
 ┗ README.md
```

## Command to run the program.

```python
python gen_shadow.py --height image_height --width image_width
```

## Example input

![input](./example_input.png)

## Example mask

![mask](./example_mask.png)

## Example output

![output](./example_output.png)