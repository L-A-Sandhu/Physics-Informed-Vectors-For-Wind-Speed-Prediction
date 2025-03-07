import numpy as np
import math

class PhysicsModelWindError:
    def __init__(self, X_data):
        self.X_data = X_data
        print("\n=== Data Summary ===")
        print(f"Total examples: {X_data.shape[0]}")
        print(f"Time steps: {X_data.shape[1]}")
        print(f"Features: {X_data.shape[2]} (0:wind_dir, 1:wind_speed, 2:pressure, 3:temp, 4:humidity)")
        print(f"Locations: {X_data.shape[3]} (0:center, 1:north, ..., 8:southwest)")
        
        # Verify data variation
        print("\n=== Data Variation Check ===")
        print("Wind speed stats (all data):")
        print(f"Min={np.min(X_data[:,:,:]):.4f}, Max={np.max(X_data[:,:,:]):.4f}, Mean={np.mean(X_data[:,:,:]):.4f}")
        
        self.locations = {
            'center': 0, 'north': 1, 'south': 2, 'east': 3, 'west': 4,
            'northeast': 5, 'northwest': 6, 'southeast': 7, 'southwest': 8
        }
        self.num_examples = X_data.shape[0]
        
    def _compute_navier_term(self, data_t):
        pressure = data_t[2, :]
        wind_speeds = data_t[1, :]
        center_wind = wind_speeds[self.locations['center']]
        
        # Pressure gradients
        east_p = pressure[self.locations['east']]
        west_p = pressure[self.locations['west']]
        grad_p_x = (east_p - west_p)/2.0
        
        north_p = pressure[self.locations['north']]
        south_p = pressure[self.locations['south']]
        grad_p_y = (north_p - south_p)/2.0
        
        # Viscosity term
        neighbor_sum = np.sum(wind_speeds) - center_wind
        laplacian = neighbor_sum - 8 * center_wind
        
        print(f"\nNavier Term Components:")
        print(f"East pressure: {east_p:.4f}, West pressure: {west_p:.4f} => grad_p_x: {grad_p_x:.4f}")
        print(f"North pressure: {north_p:.4f}, South pressure: {south_p:.4f} => grad_p_y: {grad_p_y:.4f}")
        print(f"Neighbor wind sum: {neighbor_sum:.4f}, Laplacian: {laplacian:.4f}")
        
        return (-grad_p_x - grad_p_y + 0.01 * laplacian)
    
    def _compute_advection_term(self, data_t):
        wind_dir_norm = data_t[0, self.locations['center']]
        wind_speed = data_t[1, self.locations['center']]
        wind_dir_deg = wind_dir_norm * 360.0
        theta = math.radians(wind_dir_deg)
        
        u = wind_speed * math.cos(theta)
        v = wind_speed * math.sin(theta)
        
        east_wind = data_t[1, self.locations['east']]
        west_wind = data_t[1, self.locations['west']]
        grad_x = (east_wind - west_wind)/2.0
        
        north_wind = data_t[1, self.locations['north']]
        south_wind = data_t[1, self.locations['south']]
        grad_y = (north_wind - south_wind)/2.0
        
        print(f"\nAdvection Term Components:")
        print(f"Wind dir: {wind_dir_deg:.1f}°, Speed: {wind_speed:.4f}")
        print(f"u={u:.4f}, v={v:.4f}")
        print(f"East wind: {east_wind:.4f}, West wind: {west_wind:.4f} => grad_x: {grad_x:.4f}")
        print(f"North wind: {north_wind:.4f}, South wind: {south_wind:.4f} => grad_y: {grad_y:.4f}")
        
        return -(u * grad_x + v * grad_y)
    
    def compute_errors(self, mode='NA'):
        example_errors = {'N': [], 'A': [], 'NA': []}
        
        for i in range(min(2, self.num_examples)):  # Only check first 2 examples
            print(f"\n\n=== Processing Example {i} ===")
            errors_N = []
            errors_A = []
            errors_NA = []
            
            for t in range(3, 5):  # Only check first 2 time steps
                print(f"\n-- Hour {t} to {t+1} --")
                data_t = self.X_data[i, t]
                current_wind = data_t[1, self.locations['center']]
                target = self.X_data[i, t+1, 1, self.locations['center']]
                
                print(f"Current center wind: {current_wind:.4f}")
                print(f"Next hour target: {target:.4f}")
                
                delta_N = 0
                delta_A = 0
                
                if mode in ['N', 'NA']:
                    delta_N = self._compute_navier_term(data_t)
                    print(f"Navier delta: {delta_N:.6f}")
                
                if mode in ['A', 'NA']:
                    delta_A = self._compute_advection_term(data_t)
                    print(f"Advection delta: {delta_A:.6f}")
                
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
                
                print(f"Current error: {error:.6f}" if mode != 'NA' else f"NA errors: {error_N:.6f}, {error_A:.6f}")
            
            # Store average errors
            if mode == 'N':
                avg_error = np.mean(errors_N)
                example_errors['N'].append(avg_error)
                print(f"\nExample {i} Avg N Error: {avg_error:.6f}")
            elif mode == 'A':
                avg_error = np.mean(errors_A)
                example_errors['A'].append(avg_error)
                print(f"\nExample {i} Avg A Error: {avg_error:.6f}")
            elif mode == 'NA':
                avg_error = np.mean(errors_NA)
                example_errors['NA'].append(avg_error)
                print(f"\nExample {i} Avg NA Error: {avg_error:.6f}")
        
        return example_errors

# Test with synthetic data
print("\n\n=== Synthetic Data Test ===")
synthetic_X = np.random.rand(2, 24, 5, 9)  # Random normalized data
model = PhysicsModelWindError(synthetic_X)
model.compute_errors('NA')