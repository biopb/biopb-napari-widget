# biopb-napari-widget

[napari](https://napari.org) widgets for [biopb](https://biopb.org):

- **Tensor Browser**: browse the images a biopb data server holds, and open
  them in napari as multiscale layers that load on demand.
- **OME-Zarr writers**: *File > Save Selected Layers* to a standalone OME-Zarr.

## Install

```sh
pip install biopb-napari-widget
```

Then open the widgets from napari's *Plugins* menu.

## Connecting

The Tensor Browser finds the data server that biopb runs on this machine
(`biopb control start`). Without one, it asks for a server address and token.

## License

MIT
