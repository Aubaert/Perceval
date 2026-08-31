# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ..abstract_config import AbstractRemoteConfig

TOKEN_ENV_VAR = "SCALEWAY_CLOUD_TOKEN"

PROVIDER_KEY = "provider_name"


class ScalewayConfig(AbstractRemoteConfig):
    """Handle the remote configuration for the Scaleway API.

    Note: the ``secret_key`` argument from the Scaleway classes is stored under the name ``token`` for coherence with
    the other provider configs.

    :param persistent_data: The persistent data access to use. In a standard environment, always use the default.
    """

    _token_env_var = TOKEN_ENV_VAR

    _REMOTE_KEY = "scaleway"
    _DEFAULT_URL = "https://api.scaleway.com"
    _DEFAULT_PLATFORM_PROVIDER = "quandela"

    _provider_name: str = None

    _FIELDS = AbstractRemoteConfig._FIELDS | {
        PROVIDER_KEY: "_provider_name",
    }

    @classmethod
    def set_provider(cls, provider_name: str) -> None:
        """Set a provider name in the configuration cache. It is not saved on disk before the `save` method
        is called.

        :param provider_name: The provider to use by default
        """
        cls._provider_name = provider_name

    def get_provider(self) -> str:
        """Search a valid provider from the environment, put it in cache and return it.

        The priority for the provider search is as follows:
        * A provider already in cache (e.g. set by the user or already found in a previous call)
        * The value in Perceval persistent configuration

        :return: The stored provider
        """
        if not self._provider_name:
            self.set_provider(self._get_remote_config(PROVIDER_KEY))
        return self._provider_name or self._DEFAULT_PLATFORM_PROVIDER
