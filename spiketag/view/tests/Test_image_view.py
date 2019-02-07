from spiketag.view import image_view
from vispy import scene, app


def get_test_image():
    """Load an image from the demo-data repository if possible. Otherwise,
    just return a randomly generated image.
    """
    from vispy.io import load_data_file, read_png
    
    try:
        return read_png(load_data_file('mona_lisa/mona_lisa_sm.png'))
    except Exception as exc:
        # fall back to random image
        print("Error loading demo image data: %r" % exc)

    # generate random image
    image = np.random.normal(size=(100, 100, 3))
    image[20:80, 20:80] += 3.
    image[50] += 3.
    image[:, 50] += 3.
    image = ((image - image.min()) *
             (253. / (image.max() - image.min()))).astype(np.ubyte)
    return image


if __name__ == '__main__':
    img = get_test_image()
    print(img.shape)
    im_view = image_view()
    im_view.set_data(img)
    im_view.run()