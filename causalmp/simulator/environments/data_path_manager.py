from pathlib import Path
from importlib import resources

class DataPathManager:
    @staticmethod
    def get_environment_data_path(environment_type: str, filename: str) -> Path:
        """Get path to environment data file."""
        data_path = Path(resources.files('causalmp')) / 'simulator' / 'environments' / 'environments_base_data'
        return data_path / environment_type / filename

    @staticmethod
    def validate_data_paths():
        """Validate that all required data paths exist."""
        required_dirs = [
            'belief_adoption_model',
            'NYC_taxi_routes',
            'exercise_encouragement_program'
        ]
        data_path = Path(resources.files('causalmp')) / 'simulator' / 'environments' / 'environments_base_data'
        
        for dir_name in required_dirs:
            if not (data_path / dir_name).exists():
                raise FileNotFoundError(f"Required data directory not found: {dir_name}")