#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
import streamlit as st
import base64

# In[17]:


def load_points_from_csv(file_path):
    """Load points from a CSV file using P,N,E,Z,D format."""
    df = pd.read_csv(file_path, header=None)
    df.columns = ['ID', 'Northing', 'Easting', 'Elevation', 'Description']
    return df


# In[18]:


def transform_points(params, field_points):
    """Apply a 2D transformation (scaling, rotation, translation) to field points."""
    scale, angle, tx, ty = params
    transformed = np.empty_like(field_points)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    for i, (x, y) in enumerate(field_points):
        x_scaled, y_scaled = x * scale, y * scale
        x_new = x_scaled * cos_angle - y_scaled * sin_angle + tx
        y_new = x_scaled * sin_angle + y_scaled * cos_angle + ty
        transformed[i] = [x_new, y_new]

    return transformed


# In[19]:


def compute_transformation_params(control_points, field_points):
    """Find transformation parameters that best fit field points to control points."""
    initial_params = [1, 0, 0, 0]  # Initial guess for scale, rotation, tx, ty

    def error_function(params):
        transformed_points = transform_points(params, field_points)
        return np.sum((transformed_points - control_points)**2)

    result = minimize(error_function, initial_params, method='L-BFGS-B')
    return result.x


# In[20]:


def match_common_points(control_points, field_points):
    """Match common points based on ID."""
    common_field_points = field_points[field_points['ID'].isin(control_points['ID'])]
    common_control_points = control_points[control_points['ID'].isin(field_points['ID'])]
    return common_control_points, common_field_points

def remove_outliers(initial_transformed_points, control_points, percentage=25):
    """Remove a percentage of points with the highest distance errors."""
    errors = np.linalg.norm(initial_transformed_points - control_points[['Easting', 'Northing']].values, axis=1)
    error_threshold = np.percentile(errors, 100 - percentage)
    keep_indices = errors < error_threshold
    return keep_indices

def calculate_elevation_adjustment(reduced_common_control_points, reduced_common_field_points):
    """Calculate the average delta in elevation for adjustment."""
    elevation_delta = reduced_common_control_points['Elevation'] - reduced_common_field_points['Elevation']
    return elevation_delta.mean()

def apply_elevation_adjustment(field_points, elevation_adjustment):
    """Apply elevation adjustment to all field points."""
    field_points['Adjusted Elevation'] = field_points['Elevation'] + elevation_adjustment
    return field_points


# In[21]:


def create_adjusted_points(control_points, field_points):
    common_control_points, common_field_points = match_common_points(control_points, field_points)

    if len(control_points) == 0 or len(field_points) == 0:
        # Return an empty DataFrame if there's no data to process
        return pd.DataFrame(columns=['ID', 'Adjusted Northing', 'Adjusted Easting', 'Adjusted Elevation', 'Description']
        )

    # Initial best-fit transformation
    initial_params = compute_transformation_params(common_control_points[['Easting', 'Northing']].values, 
                                                   common_field_points[['Easting', 'Northing']].values)
    initial_transformed = transform_points(initial_params, common_field_points[['Easting', 'Northing']].values)

    # Remove outliers
    keep_indices = remove_outliers(initial_transformed, common_control_points)
    reduced_common_field_points = common_field_points.iloc[keep_indices]
    reduced_common_control_points = common_control_points.iloc[keep_indices]

    # Final transformation with reduced set
    final_params = compute_transformation_params(reduced_common_control_points[['Easting', 'Northing']].values, 
                                                 reduced_common_field_points[['Easting', 'Northing']].values)
    adjusted_points = transform_points(final_params, field_points[['Easting', 'Northing']].values)
    
    # Calculate and apply elevation adjustment
    elevation_adjustment = calculate_elevation_adjustment(reduced_common_control_points, reduced_common_field_points)
    adjusted_field_points = apply_elevation_adjustment(field_points, elevation_adjustment)

  # Apply final transformation to adjusted field points (with elevation)
    adjusted_points = transform_points(final_params, field_points[['Northing', 'Easting']].values)
    adjusted_points_df = pd.DataFrame(adjusted_points, columns=['Adjusted Northing', 'Adjusted Easting'], index=field_points['ID'])
    adjusted_points_df['Adjusted Elevation'] = field_points['Adjusted Elevation']
    adjusted_points_df['Description'] = field_points['Description']  # Retain the description

    # Rounding to 3 decimal places
    adjusted_points_df = adjusted_points_df.round(3)3)
# Reset index to add 'ID' as a column and reorder the columns
    adjusted_points_df.reset_index(inplace=True)
    adjusted_points_df = adjusted_points_df[['ID', 'Adjusted Northing', 'Adjusted Easting', 'Adjusted Elevation', 'Description']]

    return adjusted_points_dfoints_df

# In[22]:


def main():
    st.title("GCP Trasnformation Application")

    # File upload widgets (if you are using file upload in Streamlit)
    control_file = st.file_uploader("Upload Control Points CSV", type=['csv'])
    field_file = st.file_uploader("Upload Field Points CSV", type=['csv'])

    if control_file and field_file:
        control_points = pd.read_csv(control_file, header=None)
        field_points = pd.read_csv(field_file, header=None)
        control_points.columns = ['ID', 'Northing', 'Easting', 'Elevation']
        field_points.columns = ['ID', 'Northing', 'Easting', 'Elevation']

        # Debugging: Print the dataframes
        st.write("Control Points:")
        st.write(control_points)
        st.write("Field Points:")
        st.write(field_points)

        # Call to create_adjusted_poadjusted_points = create_adjusted_points(control_points, field_points)

        if adjusted_points is not None and not adjusted_points.empty:
            st.write("Adjusted Points:")
            st.write(adjusted_points)usted_points)

            # Convert Dat and provide a download linkaFrame to CSVcsv = adjusted_points.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="adjusted_points.csv">Download Adjusted Points CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("No adjusted points to display.")

if __name__ == "__main__":
    main()":
    main()

