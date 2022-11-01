# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup the nasbench library.

This file is automatically run as part of `pip install -e .`
"""

import setuptools
from pkg_resources import DistributionNotFound, get_distribution


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

kw = {
    'install_requires': []
}

if get_dist('tensorflow') is None:
    kw['install_requires'].append('tensorflow-macos')
else:
    kw['install_requires'].append('tensorflow')

setuptools.setup(
    name='nasbench',
    version='1.0',
    packages=setuptools.find_packages(),
    **kw
)
