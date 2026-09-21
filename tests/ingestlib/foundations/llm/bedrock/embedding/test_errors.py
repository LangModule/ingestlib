"""Client-side validation guards in embed_text / embed_image."""
import pytest

from ingestlib.foundations.llm.bedrock.embedding import embed_image, embed_text


def test_invalid_text_dimension_raises_value_error():
    with pytest.raises(ValueError, match="dimension"):
        embed_text("hello", dimension=512)


def test_invalid_image_dimension_raises_value_error(photo_path):
    with pytest.raises(ValueError, match="dimension"):
        embed_image(photo_path.read_bytes(), format="jpeg", dimension=512)
