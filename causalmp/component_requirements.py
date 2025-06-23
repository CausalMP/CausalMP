"""Component requirements for causalmp package."""

# Define component requirements with dependencies
COMPONENT_REQUIREMENTS = {
    'estimator': {
        'dependencies': {
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'scikit-learn': '1.0.0'
        },
        'component_dependencies': []
    },
    'simulator': {
        'dependencies': {
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'scipy': '1.7.0'
        },
        'component_dependencies': []
    },
    'runner': {
        'dependencies': {
            'matplotlib': '3.4.0',
            'seaborn': '0.12.0'
        },
        'component_dependencies': ['estimator', 'simulator']
    },
    'all': {
        'dependencies': {},
        'component_dependencies': ['estimator', 'simulator', 'runner']
    }
}

def _check_version(package_name, min_version):
    """Check if a package meets the minimum version requirement."""
    try:
        # Handle scikit-learn specially
        import_name = package_name
        if package_name == 'scikit-learn':
            import_name = 'sklearn'
        
        import importlib
        pkg = importlib.import_module(import_name)
        pkg_version = pkg.__version__
        
        # Use packaging for proper version comparison
        try:
            from packaging import version
            installed = version.parse(pkg_version)
            required = version.parse(min_version)
            return installed >= required, pkg_version
        except ImportError:
            # Simple string comparison fallback
            return pkg_version >= min_version, pkg_version
    except (ImportError, AttributeError) as e:
        print(f"Debug: Error checking {package_name}: {e}")
        return False, None

def _check_installation():
    """Check which components are installed.
    
    Returns
    -------
    dict
        Dictionary with status for each component
    """
    components = {
        component: {
            'installed': True,
            'version': None,
            'missing': [],
            'outdated': []
        }
        for component in COMPONENT_REQUIREMENTS
    }

    # Check each component's dependencies
    for component, info in COMPONENT_REQUIREMENTS.items():
        # Check direct dependencies
        for package, min_version in info['dependencies'].items():
            meets_req, current_version = _check_version(package, min_version)
            if not meets_req:
                if current_version is None:
                    components[component]['installed'] = False
                    components[component]['missing'].append(
                        f"{package} (>= {min_version})"
                    )
                else:
                    components[component]['outdated'].append(
                        f"{package} ({current_version} < {min_version})"
                    )
            else:
                components[component]['version'] = current_version
        
        # Check dependencies of component dependencies
        for dep_component in info['component_dependencies']:
            for package, min_version in COMPONENT_REQUIREMENTS[dep_component]['dependencies'].items():
                meets_req, current_version = _check_version(package, min_version)
                if not meets_req:
                    if current_version is None:
                        components[component]['installed'] = False
                        components[component]['missing'].append(
                            f"{package} (>= {min_version})"
                        )
                    else:
                        components[component]['outdated'].append(
                            f"{package} ({current_version} < {min_version})"
                        )
                
    # Add 'all' component status
    all_missing = []
    all_outdated = []
    for comp_info in components.values():
        all_missing.extend(comp_info['missing'])
        all_outdated.extend(comp_info['outdated'])
    
    components['all'] = {
        'installed': len(all_missing) == 0 and len(all_outdated) == 0,
        'version': None,
        'missing': all_missing,
        'outdated': all_outdated
    }
    
    return components

def verify_component_requirements(component_name=None):
    """
    Verify that the requirements for a component are installed.
    
    Args:
        component_name (str, optional): Name of the component to check.
            If None, check all components.
    
    Returns:
        dict: Dictionary with component status information.
    """
    results = {}
    
    # Determine which components to check
    components_to_check = [component_name] if component_name else COMPONENT_REQUIREMENTS.keys()
    
    for component in components_to_check:
        if component not in COMPONENT_REQUIREMENTS:
            results[component] = {
                'installed': False,
                'version': None,
                'missing': [f"Unknown component: {component}"],
                'outdated': []
            }
            continue
        
        info = COMPONENT_REQUIREMENTS[component]
        missing = []
        outdated = []
        versions = {}
        
        # Check direct dependencies
        for pkg, ver in info['dependencies'].items():
            meets_req, detected_ver = _check_version(pkg, ver)
            if not meets_req:
                if detected_ver is None:
                    missing.append(f"{pkg} (>= {ver})")
                else:
                    outdated.append(f"{pkg} (>= {ver}, found {detected_ver})")
            versions[pkg] = detected_ver
        
        # Check component dependencies
        for dep_component in info.get('component_dependencies', []):
            if dep_component not in COMPONENT_REQUIREMENTS:
                missing.append(f"Unknown component dependency: {dep_component}")
                continue
            
            dep_result = verify_component_requirements(dep_component)
            if not dep_result[dep_component]['installed']:
                missing.append(f"Component: {dep_component}")
        
        # Determine overall status
        installed = not missing and not outdated
        
        # For the 'all' component, we need to check all other components
        if component == 'all':
            version_val = None
        else:
            # Use the version of the main package for the component
            main_pkg = next(iter(info['dependencies'].keys()), None)
            version_val = versions.get(main_pkg) if main_pkg else None
        
        results[component] = {
            'installed': installed,
            'version': version_val,
            'missing': missing,
            'outdated': outdated
        }
    
    return results