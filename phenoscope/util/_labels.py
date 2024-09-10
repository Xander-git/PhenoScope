# ----- PhenoScope Variable Label Names -----

METADATA_LABELS = [
    (STATUS_VALIDITY_LABEL := "status_valid_analysis"),
    (PLATE_NAME_LABEL := "plate_name")
]

NUMERIC_METADATA_LABELS = [
    (SAMPLING_DAY_LABEL := "SamplingDay"),
]

PHENOSCOPE_MEASUREMENTS = [
    "Intensity_IntegratedColorIntensityRed",
    "Intensity_IntegratedColorIntensityGreen",
    "Intensity_IntegratedColorIntensityBlue",
    "ImgHeight",
    "ImgWidth"
]

BASIC_CP_API_MEASUREMENT_LABELS = [
    "AreaShape",
    "Intensity",
    "Texture"
]