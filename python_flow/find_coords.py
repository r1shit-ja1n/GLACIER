import rasterio

# Apni nayi TIFF file ka naam yahan dalo
tiff_file = "output_hh.tif" 

# Google maps wale coordinates yahan dalo
latitude = 27.9125   # Example: South Lhonak Lake
longitude = 88.1950

print(f"Reading {tiff_file}...")

try:
    with rasterio.open(tiff_file) as src:
        # src.index() real-world GPS ko Pixel Row/Col mein convert karta hai
        row, col = src.index(longitude, latitude)
        
        print("\n=== MIL GAYE PIXEL COORDINATES ===")
        print(f"Glacier X (glacier_x): {col}")
        print(f"Glacier Y (glacier_y): {row}")
        print("==================================\n")
        
        print("Ye dono values apne 'simulate_glof_inundation' function mein daal do!")

except Exception as e:
    print(f"Error: {e}")
    print("Shayad tumhari TIFF file ka CRS (Coordinate System) alag hai.")