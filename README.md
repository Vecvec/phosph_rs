 # Phosph-rs

**Work in Progress**

A for fun path-tracing library.

This library provides some path-tracing shaders and an API to interact with them using `wgpu`'s 
ray-tracing extensions.

## API Design
###### Subject to change - Feedback welcome
This API is designed to provide raytracing shaders, but as little as possible else while still
keeping interactions with the shaders easy to allow other tools around this.

## API Usage
### General Usage Notes
Path `path_tracing` provides path-tracing shaders while path `debug` provides debugging shaders.

Call `pipeline_layout` with the device and the number of BLASes to create a generic pipeline
layout compatible with all shaders provided by this.

`Material` struct is for materials call `Material::new` with the texture coordinates for the three
triangles, the texture index, the maximum emission brightness an optional refractive index, and a type,
call `descriptor` on a slice of `Material`s.

***Do not*** treat the shader qualities as your main quality / performance option, even when changing
samples by the recommended changes, these do not change performance consistently.

### Texture Loading
Call `TextureLoader::new()` or `TextureLoader::default()` to create a blank texture loader. Call 
either `texture_loader.load` with a diffuse texture file name and optional lit and attribute texture
file names, or call `texture_loader.load_from_bytes` with the bytes of a diffuse texture, and optionally
bytes of lit and attribute textures as well.

To get the raw textures out of the `TextureLoader` call `texture_loader.raw` which will consume this
texture loader and output a struct containing the bytes of 3 cuboid textures (designed for wgpu's
`texture_2d_array`) that are the diffuse texture array, the lit texture array, and the attributes
texture array, this struct also contains the sizes for these three texture maps and any textures not
included. Instead, to load the textures into wgpu textures call `texture_loader.create_texture` which
will write these into wgpu textures and return a struct called `WgpuTextures` that can have `as_bind_group`
called to convert it into a bind group, and it's bind group layout.

Note: the attribute texture has the red color channel for roughness, and the other color channels are unused

### Camera
Call `Camera::new` with projection and view matrices or construct it with inverted projection
and view matrices, call `descriptor` to get a buffer descriptor.

### Roughness
This library supports four components in an attributes texture (x and y for the base material and z and w for
a transparent surface in the same place)

## Goal

Make a real time (20 - 30fps) path-tracer capable of being integrated into other renderers (yes, this is a lofty goal).
Currently, reasonable picture quality (in an artificially lit environment - cornell box test scene, just outdoors higher
quality) is only really possible with the `Medium` shader at ~256 samples per pixel with denoising (which currently gets
~4 fps). 

## Examples

Currently, this has one example: the cornell box scene. This can be run using either without a denoiser or with a denoiser
by enabling the feature `denoise` which uses oidn (you may have to do some copying of oidn dlls if using this).

## Features in development

### Importance Sampling
Note that this is WIP and not functional