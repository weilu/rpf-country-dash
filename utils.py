from shapely.geometry import shape, MultiPolygon, Polygon


def filter_country_sort_year(df, country):
    """
    Preprocess the dataframe to filter by country and sort by year
    :param df: DataFrame
    :param country: str
    :return: DataFrame
    """
    df = df.loc[df["country_name"] == country]
    df = df.sort_values(["year"])
    return df


def filter_geojson_by_country(geojson, country):
    """Filter geojson object by country
    Params:
        geojson (GeoJSON object): input geojson
        country:  str
    """
    filtered_features = [
        feature
        for feature in geojson["features"]
        if feature["properties"].get("country") == country
    ]
    filtered_geojson = {"type": "FeatureCollection", "features": filtered_features}
    return filtered_geojson


def map_center(geojson):
    polygons = []
    for feature in geojson["features"]:
        geom = shape(feature["geometry"])
        if isinstance(geom, Polygon):  # If it's a single Polygon, add it directly
            polygons.append(geom)
        elif isinstance(
            geom, MultiPolygon
        ):  # If it's a MultiPolygon, extend the list with its polygons
            polygons.extend(geom.geoms)

    if len(polygons) > 1:
        multi_polygon = MultiPolygon(polygons)
    else:
        multi_polygon = polygons[0]

    centroid = multi_polygon.centroid
    center_lat, center_lon = (centroid.y, centroid.x)

    return {"lat": center_lat, "lon": center_lon}


zoom = {
    "Albania": 5.7,
    "Bangladesh": 5,
    "Bhutan": 6,
    "Burkina Faso": 4.7,
    "Colombia": 3.6,
    "Kenya": 4.35,
    "Mozambique": 3.35,
    "Nigeria": 4.2,
    "Pakistan": 3.7,
    "Paraguay": 4.4,
    "Tunisia": 4.5,
}
