import numpy as np
import pytest

from inplace.errors import PipelineError
from inplace.sam_masks import rasterize_sam

# Official frame from the meta-sam-parser README. Lossless payload, box-local mask.
_FRAME = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)


def test_sam_text_blits_box_local_masks_and_leaves_gaps_empty() -> None:
    later = _FRAME.replace("<0f>", "<2f>", 1)
    masks = rasterize_sam(_FRAME + later, frame_count=3, width=320, height=334)
    assert len(masks) == 3
    assert masks[0].shape == (334, 320)
    assert masks[0][:228, :].max() == 0
    assert masks[0][:, :155].max() == 0
    assert masks[0][228:254, 211:270].max() > 0
    assert masks[0][228:254, 155:202].max() > 0
    assert masks[1].max() == 0
    assert masks[2].max() > 0
    assert masks[0].dtype == np.uint8


def test_stride_keeps_every_nth_frame() -> None:
    later = _FRAME.replace("<0f>", "<2f>", 1)
    masks = rasterize_sam(_FRAME + later, frame_count=4, width=320, height=334, stride=2)
    assert len(masks) == 2
    assert masks[0].max() > 0
    assert masks[1].max() > 0


def test_sam_frame_past_the_clip_fails() -> None:
    late = _FRAME.replace("<0f>", "<9f>", 1)
    with pytest.raises(PipelineError, match="frame 9"):
        rasterize_sam(late, frame_count=2, width=320, height=334)
