# Coastal data loader

This library is desinged to obtain data for coastal and nearshore ocean ecosystem and combine them in to a gridded data format for down stream modeling tasks. The primary goal of this code base is to load thermal remote sensing images and covarites that drive nearshore ocean temperatures to feed into high reolution sea surface temperature models. 

## Project structure. 

This package is driven by a configuration file that defines the data products  acquired, the data range and areas where they are pulled from and  options for how they are compiled. 

A project has a few key components 

- name: a name for the project 

- time: A field that defines the start and end for the data aquisition
    - fields: start date, end_date   

- grid: paramters that define the grid that the data products are mapped ot in each area of interest.
    - fields: 
        - resolution_m: resolution of the grid.       
        - target_crs: the CRS used for the aois or the method for selecting it       
        - resampling_continuous: method for resampling continuous variables, e.g. bilinear.
        - resampling_categorical: method for resampling categoraicl variables, e.g. nearest.
        - snap_origin: Is the grid aligned with the origin, default = true

- products: A list of data types to load, when available for each area of interest. Global options for each prodct are listed here as well. The avaible processes and options are described in detail in the following section. 
-  regions: defines the locations to obatian data for using a hierarchical structure. The base unit is an area of interst which is defined by a bounding box. Data aproducts are obtained for each AOI and projected to a unique grid defined over the AOI. The next uity up is the region. Regions are a larger spatial unit which is intended to group aois that use common data sources. some data products like digital elevation models are only avaible for specific regions. If a project need to use multiple data sources to cover the full geographic extent of the study then the AOIs that have the same data sources should be grouped into a region. 


```
regions:
  - name: pnw_estuaries
    # Region-DEPENDENT source options (which source has coverage here, etc.).
    sources:
      bathymetry:
        dem_source: cudem
    # list aois in the region.
    areas:
      - name: tillamook_bay
        center_lat: 45.52
        center_lon: -123.925
        buffer_ns_km: 25
        buffer_ew_km: 15
```

## Authenticating to google earth engine and NASA earth access

## Defining data grids

all data products are mapped to a common grid within each AOI. The end product is a matching grid of observations for each variable, for each day. These matched data points can then be used to fit grid cell level models or fed into machine lenaring lagorithms like convolutional nerual networks that require a grid based structure. 

The grids are defined by a target resolution (defaults to 100m) and a cordinate reference system is chosen using the UTM zone assocaited with the longitude of the center of the AOI. 


## Data sources

### ECOSTRESS
ECOSTRESS provides the core high resolution thermal images used in the analysis. ECOSTRESS has overpasses every 1 to 5 days with overpasses occuring at differnt times of day providing a unique data set for capturing ocean temperatures under a range of conditions. We access these data thorugh the NASA earthdata cateloug. 

They have 


