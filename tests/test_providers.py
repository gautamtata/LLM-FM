import pytest

from src.tfsp.providers import AnthropicProvider, BaseProvider, OpenAIProvider


class TestBaseProvider:
    """Tests for base provider interface."""

    def test_base_provider_is_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_init_with_model(self):
        """Test initialization with custom model."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        assert provider.model == "gpt-4"

    def test_init_default_model(self):
        """Test initialization with default model."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.model == "gpt-4o-mini"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_init_with_model(self):
        """Test initialization with custom model."""
        provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
        assert provider.model == "claude-3-opus-20240229"

    def test_init_default_model(self):
        """Test initialization with default model."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.model == "claude-sonnet-4-5-20250929"

