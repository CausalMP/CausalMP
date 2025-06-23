from setuptools import setup, find_packages
import os
import sys
import importlib.util

# Add current directory to Python path for imports
sys.path.insert(0, os.path.abspath('.'))

# Load the module from the package
spec = importlib.util.spec_from_file_location(
    "component_requirements", 
    os.path.join("causalmp", "component_requirements.py")
)
component_requirements = importlib.util.module_from_spec(spec)
spec.loader.exec_module(component_requirements)
COMPONENT_REQUIREMENTS = component_requirements.COMPONENT_REQUIREMENTS

# Read requirements from requirements.txt
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the requirements
requirements = read_requirements('requirements.txt')

# Define extras_require from COMPONENT_REQUIREMENTS
extras_require = {
    component: [
        f"{pkg}>={ver}" 
        for pkg, ver in info['dependencies'].items()
    ]
    for component, info in COMPONENT_REQUIREMENTS.items()
}

# Add development extras
extras_require.update({
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.900',
        'isort>=5.9.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'sphinx-copybutton>=0.4.0'
    ]
})

# Add 'all' extra that includes everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name='causalmp',
    version='0.1.0',
    description='A Python package for an interference gym and counterfactual estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sadegh Shirani, Yuwei Luo, William Overman, Ruoxuan Xiong, Mohsen Bayati',
    author_email='causalmp@gmail.com',
    url='https://github.com/CausalMP/CausalMP',
    packages=find_packages(include=['causalmp', 'causalmp.*']),
    python_requires='>=3.8',
    install_requires=[
        'importlib-metadata>=4.0.0',
        'packaging>=20.0',
        'numpy>=1.20.0', 
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ] + requirements,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords=[
        'causal inference',
        'network interference',
        'counterfactual estimation',
        'simulation',
        'machine learning',
        'data science'
    ],
    project_urls={
        'Documentation': 'https://causalmp.readthedocs.io/',
        'Source': 'https://github.com/CausalMP/CausalMP',
        'Tracker': 'https://github.com/CausalMP/CausalMP/issues',
    },
    include_package_data=True,
    package_data={
        'causalmp': [
            'simulator/environments/environments_base_data/*/*',
            'py.typed'
        ]
    },
    zip_safe=False,
)