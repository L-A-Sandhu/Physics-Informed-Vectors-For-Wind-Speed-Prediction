import numpy as np
import math

class PhysicsModelWindError:
    def __init__(self, X_data):
        self.X_data = X_data  # Shape: (num_examples, 24, 5, 9)
        self.locations = {
            'center': 0, 'north': 1, 'south': 2, 'east': 3, 'west': 4,
            'northeast': 5, 'northwest': 6, 'southeast': 7, 'southwest': 8
        }
        self.num_examples = X_data.shape[0]
        
    def _compute_navier_term(self, data_t):
        # Pressure gradient + viscosity term
        pressure = data_t[2, :]
        wind_speeds = data_t[1, :]
        center_wind = wind_speeds[self.locations['center']]
        
        # East-West pressure gradient
        grad_p_x = (pressure[self.locations['east']] - pressure[self.locations['west']])/2.0
        # North-South pressure gradient
        grad_p_y = (pressure[self.locations['north']] - pressure[self.locations['south']])/2.0
        
        # Viscosity term (laplacian approximation)
        neighbor_sum = np.sum(wind_speeds) - center_wind  # Sum of 8 neighbors
        laplacian = neighbor_sum - 8 * center_wind
        
        return (-grad_p_x - grad_p_y + 0.01 * laplacian)  # Empirical viscosity
    
    def _compute_advection_term(self, data_t):
        # Wind direction and speed at center
        wind_dir_norm = data_t[0, self.locations['center']]
        wind_speed = data_t[1, self.locations['center']]
        
        # Convert wind direction to radians (original: 0-360 degrees)
        wind_dir_deg = wind_dir_norm * 360.0
        theta = math.radians(wind_dir_deg)
        
        # Velocity components
        u = wind_speed * math.cos(theta)
        v = wind_speed * math.sin(theta)
        
        # Wind speed gradients
        grad_x = (data_t[1, self.locations['east']] - 
                 data_t[1, self.locations['west']])/2.0
        grad_y = (data_t[1, self.locations['north']] - 
                 data_t[1, self.locations['south']])/2.0
        
        return -(u * grad_x + v * grad_y)
    
    def compute_errors(self, mode='NA'):
        example_errors = {'N': [], 'A': [], 'NA': []}
        
        for i in range(self.num_examples):
            errors_N = []
            errors_A = []
            errors_NA = []
            
            # Start from 4th hour (index 3) to 23rd hour (index 22)
            for t in range(3, 23):
                data_t = self.X_data[i, t]
                current_wind = data_t[1, self.locations['center']]
                target = self.X_data[i, t+1, 1, self.locations['center']]
                
                delta_N = 0
                delta_A = 0
                
                if mode in ['N', 'NA']:
                    delta_N = self._compute_navier_term(data_t)
                if mode in ['A', 'NA']:
                    delta_A = self._compute_advection_term(data_t)
                
                # Calculate errors
                if mode == 'N':
                    error = (current_wind + delta_N - target)**2
                    errors_N.append(error)
                elif mode == 'A':
                    error = (current_wind + delta_A - target)**2
                    errors_A.append(error)
                elif mode == 'NA':
                    error_N = (current_wind + delta_N - target)**2
                    error_A = (current_wind + delta_A - target)**2
                    errors_NA.append((error_N + error_A)/2)
            
            # Average errors for this example
            if mode == 'N':
                example_errors['N'].append(np.mean(errors_N))
            elif mode == 'A':
                example_errors['A'].append(np.mean(errors_A))
            elif mode == 'NA':
                example_errors['NA'].append(np.mean(errors_NA))
        
        # Convert to numpy arrays
        for k in example_errors:
            example_errors[k] = np.array(example_errors[k])
            
        return example_errors

