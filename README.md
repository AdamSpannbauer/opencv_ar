# Learning AR stuff with Python+OpenCV

----

## `ar_overlay_2d.py` - 2D AR Overlay of a Query in a Target Image

Find a `query_image` in a `target_image` and do best to replace found query with `ar_image`.

### Video overlay

#### Inputs

`query_image`                | `target_image`                 | `ar_image`
:---------------------------:|:------------------------------:|:--------------|
<img src='images/crossword_query.png' width=200>  |  <h4>Live Webcam Feed</h4>  | <img src='images/smash_box_art.png' width=200>

#### Output

<p align='center'>
  <img src='readme/crossword_replace.gif' width=350>
</p>


### Static image overlay

#### Inputs

`query_image`                | `target_image`                 | `ar_image`
:---------------------------:|:------------------------------:|:--------------|
<img src='images/book_query_image.png' width=200>  |  <img src='images/book_target_image.png' width=250>  | <img src='images/smash_box_art.png' width=200>

#### Output

<p align='center'>
  <img src='readme/example.png' width=350>
</p>
