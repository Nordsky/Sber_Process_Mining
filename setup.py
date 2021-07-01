import setuptools
import re

version_file = "sberpm/_version.py"


def get_version(version_file):
    version_line = open(version_file, "rt").read()
    pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"
    result = re.search(pattern, version_line, re.M)
    if result:
        return result.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {version_file}.")


def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setuptools.setup(
    name="sberpm",
    version=get_version(version_file),
    author="Sber Process Mining R&D Team",
    author_email='Smetanev.d.m@sberbank.ru',
    description="Python Library for Process Mining",
    url="https://github.com/SberProcessMining/Sber_Process_Mining",
    include_package_data=True,
    packages=['sberpm',
              'sberpm.autoinsights',
              'sberpm.bpmn', 'sberpm.bpmn._bpmn_graph_to_file',
              'sberpm.metrics',
              'sberpm.miners',
              'sberpm.ml', 'sberpm.ml.processes', 'sberpm.ml.vectorizer',
              'sberpm.ml.reinforcement_learning',
              'sberpm.visual'
              ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    install_requires=parse_requirements('requirements.txt')
)
