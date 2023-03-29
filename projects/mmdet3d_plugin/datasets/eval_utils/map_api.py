# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# + Map mask by Kiwoo Shin, 2019.
# + Methods operating on NuScenesMap and NuScenes by Holger Caesar, 2019.

import json
import os
import random
from typing import Dict, List, Tuple, Optional, Union

import cv2
import math
import descartes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Arrow
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from tqdm import tqdm

from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from functools import partial

# Recommended style to use as the plots will show grids.
plt.style.use('seaborn-whitegrid')

# Define a map geometry type for polygons and lines.
Geometry = Union[Polygon, LineString]

locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']


class NuScenesMap:
    """
    NuScenesMap database class for querying and retrieving information from the semantic maps.
    Before using this class please use the provided tutorial `map_expansion_tutorial.ipynb`.

    Below you can find the map origins (south western corner, in [lat, lon]) for each of the 4 maps in nuScenes:
    boston-seaport: [42.336849169438615, -71.05785369873047]
    singapore-onenorth: [1.2882100868743724, 103.78475189208984]
    singapore-hollandvillage: [1.2993652317780957, 103.78217697143555]
    singapore-queenstown: [1.2782562240223188, 103.76741409301758]

    The dimensions of the maps are as follows ([width, height] in meters):
    singapore-onenorth: [1585.6, 2025.0]
    singapore-hollandvillage: [2808.3, 2922.9]
    singapore-queenstown: [3228.6, 3687.1]
    boston-seaport: [2979.5, 2118.1]
    The rasterized semantic maps (e.g. singapore-onenorth.png) published with nuScenes v1.0 have a scale of 10px/m,
    hence the above numbers are the image dimensions divided by 10.

    We use the same WGS 84 Web Mercator (EPSG:3857) projection as Google Maps/Earth.
    """
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        """
        Loads the layers, create reverse indices and shortcuts, initializes the explorer class.
        :param dataroot: Path to the layers in the form of a .json file.
        :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`,
        `boston-seaport` that we want to load.
        """
        assert map_name in locations, 'Error: Unknown map name %s!' % map_name

        self.dataroot = dataroot
        self.map_name = map_name

        self.geometric_layers = ['polygon', 'line', 'node']

        # These are the non-geometric layers which have polygons as the geometric descriptors.
        self.non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']

        # We want to be able to search for lane connectors, but not render them.
        self.lookup_polygon_layers = self.non_geometric_polygon_layers + ['lane_connector']

        # These are the non-geometric layers which have line strings as the geometric descriptors.
        self.non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']
        self.non_geometric_layers = self.non_geometric_polygon_layers + self.non_geometric_line_layers
        self.layer_names = self.geometric_layers + self.lookup_polygon_layers + self.non_geometric_line_layers

        # Load the selected map.
        self.json_fname = os.path.join(self.dataroot, 'maps', 'expansion', '{}.json'.format(self.map_name))
        with open(self.json_fname, 'r') as fh:
            self.json_obj = json.load(fh)

        # Parse the map version and print an error for deprecated maps.
        if 'version' in self.json_obj:
            self.version = self.json_obj['version']
        else:
            self.version = '1.0'
        if self.version < '1.3':
            raise Exception('Error: You are using an outdated map version (%s)! '
                            'Please go to https://www.nuscenes.org/download to download the latest map!')

        self.canvas_edge = self.json_obj['canvas_edge']
        self._load_layers()
        self._make_token2ind()
        self._make_shortcuts()

        self.explorer = NuScenesMapExplorer(self)

    def _load_layer(self, layer_name: str) -> List[dict]:
        """
        Returns a list of records corresponding to the layer name.
        :param layer_name: Name of the layer that will be loaded.
        :return: A list of records corresponding to a layer.
        """
        return self.json_obj[layer_name]

    def _load_layer_dict(self, layer_name: str) -> Dict[str, Union[dict, list]]:
        """
        Returns a dict of records corresponding to the layer name.
        :param layer_name: Name of the layer that will be loaded.
        :return: A dict of records corresponding to a layer.
        """
        return self.json_obj[layer_name]

    def _load_layers(self) -> None:
        """ Loads each available layer. """

        # Explicit assignment of layers are necessary to help the IDE determine valid class members.
        self.polygon = self._load_layer('polygon')
        self.line = self._load_layer('line')
        self.node = self._load_layer('node')
        self.drivable_area = self._load_layer('drivable_area')
        self.road_segment = self._load_layer('road_segment')
        self.road_block = self._load_layer('road_block')
        self.lane = self._load_layer('lane')
        self.ped_crossing = self._load_layer('ped_crossing')
        self.walkway = self._load_layer('walkway')
        self.stop_line = self._load_layer('stop_line')
        self.carpark_area = self._load_layer('carpark_area')
        self.road_divider = self._load_layer('road_divider')
        self.lane_divider = self._load_layer('lane_divider')
        self.traffic_light = self._load_layer('traffic_light')

        self.arcline_path_3: Dict[str, List[dict]] = self._load_layer_dict('arcline_path_3')
        self.connectivity: Dict[str, dict] = self._load_layer_dict('connectivity')
        self.lane_connector = self._load_layer('lane_connector')

    def _make_token2ind(self) -> None:
        """ Store the mapping from token to layer index for each layer. """
        self._token2ind = dict()
        for layer_name in self.layer_names:
            self._token2ind[layer_name] = dict()

            for ind, member in enumerate(getattr(self, layer_name)):
                self._token2ind[layer_name][member['token']] = ind

    def _make_shortcuts(self) -> None:
        """ Makes the record shortcuts. """

        # Makes a shortcut between non geometric records to their nodes.
        for layer_name in self.non_geometric_polygon_layers:
            if layer_name == 'drivable_area':  # Drivable area has more than one geometric representation.
                pass
            else:
                for record in self.__dict__[layer_name]:
                    polygon_obj = self.get('polygon', record['polygon_token'])
                    record['exterior_node_tokens'] = polygon_obj['exterior_node_tokens']
                    record['holes'] = polygon_obj['holes']

        for layer_name in self.non_geometric_line_layers:
            for record in self.__dict__[layer_name]:
                record['node_tokens'] = self.get('line', record['line_token'])['node_tokens']

        # Makes a shortcut between stop lines to their cues, there's different cues for different types of stop line.
        # Refer to `_get_stop_line_cue()` for details.
        for record in self.stop_line:
            cue = self._get_stop_line_cue(record)
            record['cue'] = cue

        # Makes a shortcut between lanes to their lane divider segment nodes.
        for record in self.lane:
            record['left_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
                                                         record['left_lane_divider_segments']]
            record['right_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
                                                          record['right_lane_divider_segments']]

    def _get_stop_line_cue(self, stop_line_record: dict) -> List[dict]:
        """
        Get the different cues for different types of stop lines.
        :param stop_line_record: A single stop line record.
        :return: The cue for that stop line.
        """
        if stop_line_record['stop_line_type'] in ['PED_CROSSING', 'TURN_STOP']:
            return [self.get('ped_crossing', token) for token in stop_line_record['ped_crossing_tokens']]
        elif stop_line_record['stop_line_type'] in ['STOP_SIGN', 'YIELD']:
            return []
        elif stop_line_record['stop_line_type'] == 'TRAFFIC_LIGHT':
            return [self.get('traffic_light', token) for token in stop_line_record['traffic_light_tokens']]

    def get(self, layer_name: str, token: str) -> dict:
        """
        Returns a record from the layer in constant runtime.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: A single layer record.
        """
        assert layer_name in self.layer_names, "Layer {} not found".format(layer_name)

        return getattr(self, layer_name)[self.getind(layer_name, token)]

    def getind(self, layer_name: str, token: str) -> int:
        """
        This returns the index of the record in a layer in constant runtime.
        :param layer_name: Name of the layer we are interested in.
        :param token: Token of the record.
        :return: The index of the record in the layer, layer is an array.
        """
        return self._token2ind[layer_name][token]

    def render_record(self,
                      layer_name: str,
                      token: str,
                      alpha: float = 0.5,
                      figsize: Tuple[float, float] = None,
                      other_layers: List[str] = None,
                      bitmap: Optional[BitMap] = None) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
         Render a single map record. By default will also render 3 layers which are `drivable_area`, `lane`,
         and `walkway` unless specified by `other_layers`.
         :param layer_name: Name of the layer that we are interested in.
         :param token: Token of the record that you want to render.
         :param alpha: The opacity of each layer that gets rendered.
         :param figsize: Size of the whole figure.
         :param other_layers: What other layers to render aside from the one specified in `layer_name`.
         :param bitmap: Optional BitMap object to render below the other map layers.
         :return: The matplotlib figure and axes of the rendered layers.
         """
        return self.explorer.render_record(layer_name, token, alpha,
                                           figsize=figsize, other_layers=other_layers, bitmap=bitmap)

    def render_layers(self,
                      layer_names: List[str],
                      alpha: float = 0.5,
                      figsize: Union[None, float, Tuple[float, float]] = None,
                      tokens: List[str] = None,
                      bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Render a list of layer names.
        :param layer_names: A list of layer names.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_layers(layer_names, alpha,
                                           figsize=figsize, tokens=tokens, bitmap=bitmap)

    def render_map_patch(self,
                         box_coords: Tuple[float, float, float, float],
                         layer_names: List[str] = None,
                         alpha: float = 0.5,
                         figsize: Tuple[int, int] = (15, 15),
                         render_egoposes_range: bool = True,
                         render_legend: bool = True,
                         bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        return self.explorer.render_map_patch(box_coords, layer_names=layer_names, alpha=alpha, figsize=figsize,
                                              render_egoposes_range=render_egoposes_range,
                                              render_legend=render_legend, bitmap=bitmap)

    def render_map_in_image(self,
                            nusc: NuScenes,
                            sample_token: str,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 10000,
                            min_polygon_area: float = 1000,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                            layer_names: List[str] = None,
                            verbose: bool = True,
                            out_path: str = None) -> Tuple[Figure, Axes]:
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        return self.explorer.render_map_in_image(
            nusc, sample_token, camera_channel=camera_channel, alpha=alpha,
            patch_radius=patch_radius, min_polygon_area=min_polygon_area,
            render_behind_cam=render_behind_cam, render_outside_im=render_outside_im,
            layer_names=layer_names, verbose=verbose, out_path=out_path)

    def get_map_mask_in_image(self,
                              nusc: NuScenes,
                              sample_token: str,
                              camera_channel: str = 'CAM_FRONT',
                              alpha: float = 0.3,
                              patch_radius: float = 10000,
                              min_polygon_area: float = 1000,
                              render_behind_cam: bool = True,
                              render_outside_im: bool = True,
                              layer_names: List[str] = None,
                              verbose: bool = False,
                              out_path: str = None):
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        return self.explorer.get_map_mask_in_image(
            nusc, sample_token, camera_channel=camera_channel, alpha=alpha,
            patch_radius=patch_radius, min_polygon_area=min_polygon_area,
            render_behind_cam=render_behind_cam, render_outside_im=render_outside_im,
            layer_names=layer_names, verbose=verbose, out_path=out_path)

    def render_egoposes_on_fancy_map(self,
                                     nusc: NuScenes,
                                     scene_tokens: List = None,
                                     verbose: bool = True,
                                     out_path: str = None,
                                     render_egoposes: bool = True,
                                     render_egoposes_range: bool = True,
                                     render_legend: bool = True,
                                     bitmap: Optional[BitMap] = None) -> Tuple[np.ndarray, Figure, Axes]:
        """
        Renders each ego pose of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param scene_tokens: Optional list of scene tokens corresponding to the current map location.
        :param verbose: Whether to show status messages and progress bar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_egoposes: Whether to render ego poses.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: <np.float32: n, 2>. Returns a matrix with n ego poses in global map coordinates.
        """
        return self.explorer.render_egoposes_on_fancy_map(nusc, scene_tokens=scene_tokens,
                                                          verbose=verbose, out_path=out_path,
                                                          render_egoposes=render_egoposes,
                                                          render_egoposes_range=render_egoposes_range,
                                                          render_legend=render_legend, bitmap=bitmap)

    def render_centerlines(self,
                           resolution_meters: float = 0.5,
                           figsize: Union[None, float, Tuple[float, float]] = None,
                           bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Render the centerlines of all lanes and lane connectors.
        :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
            lanes are properly represented.
        :param figsize: Size of the figure.
        :param bitmap: Optional BitMap object to render below the other map layers.
        """
        return self.explorer.render_centerlines(resolution_meters=resolution_meters, figsize=figsize, bitmap=bitmap)

    def render_map_mask(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_names: List[str] = None,
                        canvas_size: Tuple[int, int] = (100, 100),
                        figsize: Tuple[int, int] = (15, 15),
                        n_row: int = 2) -> Tuple[Figure, List[Axes]]:
        """
        Render map mask of the patch specified by patch_box and patch_angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_names: A list of layer names to be returned.
        :param canvas_size: Size of the output mask (h, w).
        :param figsize: Size of the figure.
        :param n_row: Number of rows with plots.
        :return: The matplotlib figure and a list of axes of the rendered layers.
        """
        return self.explorer.render_map_mask(patch_box, patch_angle,
                                             layer_names=layer_names, canvas_size=canvas_size,
                                             figsize=figsize, n_row=n_row)

    def get_map_mask(self,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     layer_names: List[str] = None,
                     canvas_size: Optional[Tuple[int, int]] = (100, 100)) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        return self.explorer.get_map_mask(patch_box, patch_angle, layer_names=layer_names, canvas_size=canvas_size)

    def get_map_geom(self,
                     patch_box: Tuple[float, float, float, float],
                     patch_angle: float,
                     layer_names: List[str]) -> List[Tuple[str, List[Geometry]]]:
        """
        Returns a list of geometries in the specified patch_box.
        These are unscaled, but aligned with the patch angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
                            North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :return: List of layer names and their corresponding geometries.
        """
        return self.explorer.get_map_geom(patch_box, patch_angle, layer_names)

    def get_records_in_patch(self,
                             box_coords: Tuple[float, float, float, float],
                             layer_names: List[str] = None,
                             mode: str = 'intersect') -> Dict[str, List[str]]:
        """
        Get all the record token that intersects or is within a particular rectangular patch.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: Names of the layers that we want to retrieve in a particular patch. By default will always
        look at the all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch, "within" will return
        all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """
        return self.explorer.get_records_in_patch(box_coords, layer_names=layer_names, mode=mode)

    def is_record_in_patch(self,
                           layer_name: str,
                           token: str,
                           box_coords: Tuple[float, float, float, float],
                           mode: str = 'intersect') -> bool:
        """
        Query whether a particular record is in a rectangular patch
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch, "within" will
                     return True if the geometric object is within the patch.
        :return: Boolean value on whether a particular record intersects or within a particular patch.
        """
        return self.explorer.is_record_in_patch(layer_name, token, box_coords, mode=mode)

    def layers_on_point(self, x: float, y: float, layer_names: List[str] = None) -> Dict[str, str]:
        """
        Returns all the polygonal layers that a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_names: The names of the layers to search for.
        :return: All the polygonal layers that a particular point is on. {<layer name>: <list of tokens>}
        """
        return self.explorer.layers_on_point(x, y, layer_names=layer_names)

    def record_on_point(self, x: float, y: float, layer_name: str) -> str:
        """
        Query what record of a layer a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_name: The non geometric polygonal layer name that we are interested in.
        :return: The first token of a layer a particular point is on or '' if no layer is found.
        """
        return self.explorer.record_on_point(x, y, layer_name)

    def extract_polygon(self, polygon_token: str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token.
        :param polygon_token: The token of the polygon record.
        :return: The polygon wrapped in a shapely Polygon object.
        """
        return self.explorer.extract_polygon(polygon_token)

    def extract_line(self, line_token: str) -> LineString:
        """
        Construct a shapely LineString object out of a line token.
        :param line_token: The token of the line record.
        :return: The line wrapped in a LineString object.
        """
        return self.explorer.extract_line(line_token)

    def get_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the geometric object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the line representation.
        """
        return self.explorer.get_bounds(layer_name, token)

    def get_records_in_radius(self, x: float, y: float, radius: float,
                              layer_names: List[str], mode: str = 'intersect') -> Dict[str, List[str]]:
        """
        Get all the record tokens that intersect a square patch of side length 2*radius centered on (x,y).
        :param x: X-coordinate in global frame.
        :param y: y-coordinate in global frame.
        :param radius: All records within radius meters of point (x, y) will be returned.
        :param layer_names: Names of the layers that we want to retrieve. By default will always
        look at the all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch, "within" will return
        all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """

        patch = (x - radius, y - radius, x + radius, y + radius)
        return self.explorer.get_records_in_patch(patch, layer_names, mode=mode)

    def discretize_centerlines(self, resolution_meters: float) -> List[np.array]:
        """
        Discretize the centerlines of lanes and lane connectors.
        :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
            lanes are properly represented.
        :return: A list of np.arrays with x, y and z values for each point.
        """
        pose_lists = []
        for lane in self.lane + self.lane_connector:
            my_lane = self.arcline_path_3.get(lane['token'], [])
            discretized = np.array(discretize_lane(my_lane, resolution_meters))
            pose_lists.append(discretized)

        return pose_lists

    def discretize_lanes(self, tokens: List[str],
                         resolution_meters: float) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Discretizes a list of lane/lane connector tokens.
        :param tokens: List of lane and/or lane connector record tokens. Can be retrieved with
            get_records_in_radius or get_records_in_patch.
        :param resolution_meters: How finely to discretize the splines.
        :return: Mapping from lane/lane connector token to sequence of poses along the lane.
        """

        return {ID: discretize_lane(self.arcline_path_3.get(ID, []), resolution_meters) for ID in tokens}

    def _get_connected_lanes(self, lane_token: str, incoming_outgoing: str) -> List[str]:
        """
        Helper for getting the lanes connected to a given lane
        :param lane_token: Token for the lane.
        :param incoming_outgoing: Whether to get incoming or outgoing lanes
        :return: List of lane tokens this lane is connected to.
        """

        if lane_token not in self.connectivity:
            raise ValueError(f"{lane_token} is not a valid lane.")

        return self.connectivity[lane_token][incoming_outgoing]

    def get_outgoing_lane_ids(self, lane_token: str) -> List[str]:
        """
        Get the out-going lanes.
        :param lane_token: Token for the lane.
        :return: List of lane tokens that start at the end of this lane.
        """

        return self._get_connected_lanes(lane_token, 'outgoing')

    def get_incoming_lane_ids(self, lane_token: str) -> List[str]:
        """
        Get the incoming lanes.
        :param lane_token: Token for the lane.
        :return: List of lane tokens that end at the start of this lane.
        """

        return self._get_connected_lanes(lane_token, 'incoming')

    def get_arcline_path(self, lane_token: str) -> List[ArcLinePath]:
        """
        Get the arcline path representation for a lane.
        Note: This function was previously called `get_lane()`, but renamed to avoid confusion between lanes and
              arcline paths.
        :param lane_token: Token for the lane.
        :return: Arc line path representation of the lane.
        """

        arcline_path = self.arcline_path_3.get(lane_token)
        if not arcline_path:
            raise ValueError(f'Error: Lane with token {lane_token} does not have a valid arcline path!')

        return arcline_path

    def get_closest_lane(self, x: float, y: float, radius: float = 5) -> str:
        """
        Get closest lane id within a radius of query point. The distance from a point (x, y) to a lane is
        the minimum l2 distance from (x, y) to a point on the lane.
        :param x: X coordinate in global coordinate frame.
        :param y: Y Coordinate in global coordinate frame.
        :param radius: Radius around point to consider.
        :return: Lane id of closest lane within radius.
        """

        lanes = self.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']

        discrete_points = self.discretize_lanes(lanes, 0.5)

        current_min = np.inf

        min_id = ""
        for lane_id, points in discrete_points.items():

            distance = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()
            if distance <= current_min:
                current_min = distance
                min_id = lane_id

        return min_id

    def render_next_roads(self,
                          x: float,
                          y: float,
                          alpha: float = 0.5,
                          figsize: Union[None, float, Tuple[float, float]] = None,
                          bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Renders the possible next roads from a point of interest.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :param bitmap: Optional BitMap object to render below the other map layers.
        """
        return self.explorer.render_next_roads(x, y, alpha, figsize=figsize, bitmap=bitmap)

    def get_next_roads(self, x: float, y: float) -> Dict[str, List[str]]:
        """
        Get the possible next roads from a point of interest.
        Returns road_segment, road_block and lane.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :return: Dictionary of layer_name - tokens pairs.
        """
        # Filter out irrelevant layers.
        road_layers = ['road_segment', 'road_block', 'lane']
        layers = self.explorer.layers_on_point(x, y)
        rel_layers = {layer: layers[layer] for layer in road_layers}

        # Pick most fine-grained road layer (lane, road_block, road_segment) object that contains the point.
        rel_layer = None
        rel_token = None
        for layer in road_layers[::-1]:
            if rel_layers[layer] != '':
                rel_layer = layer
                rel_token = rel_layers[layer]
                break
        assert rel_layer is not None, 'Error: No suitable layer in the specified point location!'

        # Get all records that overlap with the bounding box of the selected road.
        box_coords = self.explorer.get_bounds(rel_layer, rel_token)
        intersect = self.explorer.get_records_in_patch(box_coords, road_layers, mode='intersect')

        # Go through all objects within the bounding box.
        result = {layer: [] for layer in road_layers}
        if rel_layer == 'road_segment':
            # For road segments, we do not have a direction.
            # Return objects that have ANY exterior points in common with the relevant layer.
            rel_exterior_nodes = self.get(rel_layer, rel_token)['exterior_node_tokens']
            for layer in road_layers:
                for token in intersect[layer]:
                    exterior_nodes = self.get(layer, token)['exterior_node_tokens']
                    if any(n in exterior_nodes for n in rel_exterior_nodes) \
                            and token != rel_layers[layer]:
                        result[layer].append(token)
        else:
            # For lanes and road blocks, the next road is indicated by the edge line.
            # Return objects where ALL edge line nodes are included in the exterior nodes.
            to_edge_line = self.get(rel_layer, rel_token)['to_edge_line_token']
            to_edge_nodes = self.get('line', to_edge_line)['node_tokens']
            for layer in road_layers:
                for token in intersect[layer]:
                    exterior_nodes = self.get(layer, token)['exterior_node_tokens']
                    if all(n in exterior_nodes for n in to_edge_nodes) \
                            and token != rel_layers[layer]:
                        result[layer].append(token)
        return result


class NuScenesMapExplorer:
    """ Helper class to explore the nuScenes map data. """
    def __init__(self,
                 map_api: NuScenesMap,
                 representative_layers: Tuple[str] = ('drivable_area', 'lane', 'walkway'),
                 color_map: dict = None):
        """
        :param map_api: NuScenesMap database class.
        :param representative_layers: These are the layers that we feel are representative of the whole mapping data.
        :param color_map: Color map.
        """
        # Mutable default argument.
        if color_map is None:
            color_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

        self.map_api = map_api
        self.representative_layers = representative_layers
        self.color_map = color_map

        self.canvas_max_x = self.map_api.canvas_edge[0]
        self.canvas_min_x = 0
        self.canvas_max_y = self.map_api.canvas_edge[1]
        self.canvas_min_y = 0
        self.canvas_aspect_ratio = (self.canvas_max_x - self.canvas_min_x) / (self.canvas_max_y - self.canvas_min_y)

    def render_centerlines(self,
                           resolution_meters: float,
                           figsize: Union[None, float, Tuple[float, float]] = None,
                           bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Render the centerlines of all lanes and lane connectors.
        :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
            lanes are properly represented.
        :param figsize: Size of the figure.
        :param bitmap: Optional BitMap object to render below the other map layers.
        """
        # Discretize all lanes and lane connectors.
        pose_lists = self.map_api.discretize_centerlines(resolution_meters)

        # Render connectivity lines.
        fig = plt.figure(figsize=self._get_figsize(figsize))
        ax = fig.add_axes([0, 0, 1, 1 / self.canvas_aspect_ratio])

        if bitmap is not None:
            bitmap.render(self.map_api.canvas_edge, ax)

        for pose_list in pose_lists:
            if len(pose_list) > 0:
                plt.plot(pose_list[:, 0], pose_list[:, 1])

        return fig, ax

    def render_map_mask(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_names: List[str],
                        canvas_size: Tuple[int, int],
                        figsize: Tuple[int, int],
                        n_row: int = 2) -> Tuple[Figure, List[Axes]]:
        """
        Render map mask of the patch specified by patch_box and patch_angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_names: A list of layer names to be extracted.
        :param canvas_size: Size of the output mask (h, w).
        :param figsize: Size of the figure.
        :param n_row: Number of rows with plots.
        :return: The matplotlib figure and a list of axes of the rendered layers.
        """
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        map_mask = self.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)

        # If no canvas_size is specified, retrieve the default from the output of get_map_mask.
        if canvas_size is None:
            canvas_size = map_mask.shape[1:]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, canvas_size[1])
        ax.set_ylim(0, canvas_size[0])

        n_col = len(map_mask) // n_row
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.025, hspace=0.05)
        for i in range(len(map_mask)):
            r = i // n_col
            c = i - r * n_col
            subax = plt.subplot(gs[r, c])
            subax.imshow(map_mask[i], origin='lower')
            subax.text(canvas_size[0] * 0.5, canvas_size[1] * 1.1, layer_names[i])
            subax.grid(False)

        return fig, fig.axes

    def get_map_geom(self,
                     patch_box: Tuple[float, float, float, float],
                     patch_angle: float,
                     layer_names: List[str]) -> List[Tuple[str, List[Geometry]]]:
        """
        Returns a list of geometries in the specified patch_box.
        These are unscaled, but aligned with the patch angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
                            North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :return: List of layer names and their corresponding geometries.
        """
        # If None, return all geometric layers.
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        # Get each layer name and geometry and store them in a list.
        map_geom = []
        for layer_name in layer_names:
            layer_geom = self._get_layer_geom(patch_box, patch_angle, layer_name)
            if layer_geom is None:
                continue
            map_geom.append((layer_name, layer_geom))

        return map_geom

    def map_geom_to_mask(self,
                         map_geom: List[Tuple[str, List[Geometry]]],
                         local_box: Tuple[float, float, float, float],
                         canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param map_geom: List of layer names and their corresponding geometries.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        :return: Stacked numpy array of size [c x h x w] with c channels and the same height/width as the canvas.
        """
        # Get each layer mask and stack them into a numpy tensor.
        map_mask = []
        for layer_name, layer_geom in map_geom:
            layer_mask = self._layer_geom_to_mask(layer_name, layer_geom, local_box, canvas_size)
            if layer_mask is not None:
                map_mask.append(layer_mask)

        return np.array(map_mask)

    def get_map_mask(self,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     layer_names: List[str] = None,
                     canvas_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        # For some combination of parameters, we need to know the size of the current map.
        if self.map_api.map_name == 'singapore-onenorth':
            map_dims = [1585.6, 2025.0]
        elif self.map_api.map_name == 'singapore-hollandvillage':
            map_dims = [2808.3, 2922.9]
        elif self.map_api.map_name == 'singapore-queenstown':
            map_dims = [3228.6, 3687.1]
        elif self.map_api.map_name == 'boston-seaport':
            map_dims = [2979.5, 2118.1]
        else:
            raise Exception('Error: Invalid map!')

        # If None, return the entire map.
        if patch_box is None:
            patch_box = [map_dims[0] / 2, map_dims[1] / 2, map_dims[1], map_dims[0]]

        # If None, return all geometric layers.
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        # If None, return the specified patch in the original scale of 10px/m.
        if canvas_size is None:
            map_scale = 10
            canvas_size = np.array((patch_box[2], patch_box[3])) * map_scale
            canvas_size = tuple(np.round(canvas_size).astype(np.int32))

        # Get geometry of each layer.
        map_geom = self.get_map_geom(patch_box, patch_angle, layer_names)

        # Convert geometry of each layer into mask and stack them into a numpy tensor.
        # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        map_mask = self.map_geom_to_mask(map_geom, local_box, canvas_size)
        assert np.all(map_mask.shape[1:] == canvas_size)

        return map_mask

    def render_record(self,
                      layer_name: str,
                      token: str,
                      alpha: float = 0.5,
                      figsize: Union[None, float, Tuple[float, float]] = None,
                      other_layers: List[str] = None,
                      bitmap: Optional[BitMap] = None) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Render a single map record.
        By default will also render 3 layers which are `drivable_area`, `lane`, and `walkway` unless specified by
        `other_layers`.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record that you want to render.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :param other_layers: What other layers to render aside from the one specified in `layer_name`.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        if other_layers is None:
            other_layers = list(self.representative_layers)

        for other_layer in other_layers:
            if other_layer not in self.map_api.non_geometric_layers:
                raise ValueError("{} is not a non geometric layer".format(layer_name))

        x1, y1, x2, y2 = self.map_api.get_bounds(layer_name, token)

        local_width = x2 - x1
        local_height = y2 - y1
        assert local_height > 0, 'Error: Map has 0 height!'
        local_aspect_ratio = local_width / local_height

        # We obtained the values 0.65 and 0.66 by trials.
        fig = plt.figure(figsize=self._get_figsize(figsize))
        global_ax = fig.add_axes([0, 0, 0.65, 0.65 / self.canvas_aspect_ratio])
        local_ax = fig.add_axes([0.66, 0.66 / self.canvas_aspect_ratio, 0.34, 0.34 / local_aspect_ratio])

        # To make sure the sequence of the layer overlays is always consistent after typesetting set().
        random.seed('nutonomy')

        if bitmap is not None:
            bitmap.render(self.map_api.canvas_edge, global_ax)
            bitmap.render(self.map_api.canvas_edge, local_ax)

        layer_names = other_layers + [layer_name]
        layer_names = list(set(layer_names))

        for layer in layer_names:
            self._render_layer(global_ax, layer, alpha)

        for layer in layer_names:
            self._render_layer(local_ax, layer, alpha)

        if layer_name == 'drivable_area':
            # Bad output aesthetically if we add spacing between the objects and the axes for drivable area.
            local_ax_xlim = (x1, x2)
            local_ax_ylim = (y1, y2)
        else:
            # Add some spacing between the object and the axes.
            local_ax_xlim = (x1 - local_width / 3, x2 + local_width / 3)
            local_ax_ylim = (y1 - local_height / 3, y2 + local_height / 3)

            # Draws the rectangular patch on the local_ax.
            local_ax.add_patch(Rectangle((x1, y1), local_width, local_height, linestyle='-.', color='red', fill=False,
                                         lw=2))

        local_ax.set_xlim(*local_ax_xlim)
        local_ax.set_ylim(*local_ax_ylim)
        local_ax.set_title('Local View')

        global_ax.set_xlim(self.canvas_min_x, self.canvas_max_x)
        global_ax.set_ylim(self.canvas_min_y, self.canvas_max_y)
        global_ax.set_title('Global View')
        global_ax.legend()

        # Adds the zoomed in effect to the plot.
        mark_inset(global_ax, local_ax, loc1=2, loc2=4)

        return fig, (global_ax, local_ax)

    def render_layers(self,
                      layer_names: List[str],
                      alpha: float,
                      figsize: Union[None, float, Tuple[float, float]],
                      tokens: List[str] = None,
                      bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Render a list of layers.
        :param layer_names: A list of layer names.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        fig = plt.figure(figsize=self._get_figsize(figsize))
        ax = fig.add_axes([0, 0, 1, 1 / self.canvas_aspect_ratio])

        ax.set_xlim(self.canvas_min_x, self.canvas_max_x)
        ax.set_ylim(self.canvas_min_y, self.canvas_max_y)

        if bitmap is not None:
            bitmap.render(self.map_api.canvas_edge, ax)

        layer_names = list(set(layer_names))
        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha, tokens)

        ax.legend()

        return fig, ax

    def render_map_patch(self,
                         box_coords: Tuple[float, float, float, float],
                         layer_names: List[str] = None,
                         alpha: float = 0.5,
                         figsize: Tuple[float, float] = (15, 15),
                         render_egoposes_range: bool = True,
                         render_legend: bool = True,
                         bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        x_min, y_min, x_max, y_max = box_coords

        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        fig = plt.figure(figsize=figsize)

        local_width = x_max - x_min
        local_height = y_max - y_min
        assert local_height > 0, 'Error: Map patch has 0 height!'
        local_aspect_ratio = local_width / local_height

        ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

        if bitmap is not None:
            bitmap.render(self.map_api.canvas_edge, ax)

        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha)

        x_margin = np.minimum(local_width / 4, 50)
        y_margin = np.minimum(local_height / 4, 10)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        if render_egoposes_range:
            ax.add_patch(Rectangle((x_min, y_min), local_width, local_height, fill=False, linestyle='-.', color='red',
                                   lw=2))
            ax.text(x_min + local_width / 100, y_min + local_height / 2, "%g m" % local_height,
                    fontsize=14, weight='bold')
            ax.text(x_min + local_width / 2, y_min + local_height / 100, "%g m" % local_width,
                    fontsize=14, weight='bold')

        if render_legend:
            ax.legend(frameon=True, loc='upper right')

        return fig, ax

    def render_map_in_image(self,
                            nusc: NuScenes,
                            sample_token: str,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 10000,
                            min_polygon_area: float = 1000,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                            layer_names: List[str] = None,
                            verbose: bool = True,
                            out_path: str = None) -> Tuple[Figure, Axes]:
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        near_plane = 1e-8

        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
        if layer_names is None:
            layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

        # Check layers whether we can render them.
        for layer_name in layer_names:
            assert layer_name in self.map_api.non_geometric_polygon_layers, \
                'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert self.map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (self.map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = self.get_records_in_patch(box_coords, layer_names, 'intersect')

        # Init axes.
        fig = plt.figure(figsize=(9, 16))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im_size[0])
        ax.set_ylim(0, im_size[1])
        ax.imshow(im)

        # Retrieve and render each record.
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = self.map_api.get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = self.map_api.extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                    # Remove points that are partially behind the camera.
                    depths = points[2, :]
                    behind = depths < near_plane
                    if np.all(behind):
                        continue

                    if render_behind_cam:
                        # Perform clipping on polygons that are partially behind the camera.
                        points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                    elif np.any(behind):
                        # Otherwise ignore any polygon that is partially behind the camera.
                        continue

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points) == 0 or points.shape[1] < 3:
                        continue

                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                    points = view_points(points, cam_intrinsic, normalize=True)

                    # Skip polygons where all points are outside the image.
                    # Leave a margin of 1 pixel for aesthetic reasons.
                    inside = np.ones(points.shape[1], dtype=bool)
                    inside = np.logical_and(inside, points[0, :] > 1)
                    inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                    inside = np.logical_and(inside, points[1, :] > 1)
                    inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                    if render_outside_im:
                        if np.all(np.logical_not(inside)):
                            continue
                    else:
                        if np.any(np.logical_not(inside)):
                            continue

                    points = points[:2, :]
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    polygon_proj = Polygon(points)

                    # Filter small polygons
                    if polygon_proj.area < min_polygon_area:
                        continue

                    label = layer_name
                    ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label))

        # Display the image.
        plt.axis('off')
        ax.invert_yaxis()

        if out_path is not None:
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

        return fig, ax

    @staticmethod
    def points_transform(points, poserecord, cs_record, cam_intrinsic, im_size, near_plane=1e-8,
                         render_behind_cam=True, render_outside_im=True):
        points = np.vstack((points, np.zeros((1, points.shape[1]))))

        # Transform into the ego vehicle frame for the timestamp of the image.
        points = points - np.array(poserecord['translation']).reshape((-1, 1))
        points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

        # Transform into the camera.
        points = points - np.array(cs_record['translation']).reshape((-1, 1))
        points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

        # Remove points that are partially behind the camera.
        depths = points[2, :]
        behind = depths < near_plane
        if np.all(behind):
            return None

        if render_behind_cam:
            # Perform clipping on polygons that are partially behind the camera.
            points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)

        elif np.any(behind):
            # Otherwise ignore any polygon that is partially behind the camera.
            return None

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(points, cam_intrinsic, normalize=True)

        # Skip polygons where all points are outside the image.
        # Leave a margin of 1 pixel for aesthetic reasons.
        inside = np.ones(points.shape[1], dtype=bool)
        inside = np.logical_and(inside, points[0, :] > 1)
        inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
        inside = np.logical_and(inside, points[1, :] > 1)
        inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)

        if render_outside_im:
            if np.all(np.logical_not(inside)):
                return None
        else:
            if np.any(np.logical_not(inside)):
                return None

        # points = points[:, inside]

        # Ignore polygons with less than 3 points after clipping.
        if len(points) == 0 or points.shape[1] < 3:
            return None

        points = points[:2, :]
        points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
        return points

    def get_map_mask_in_image(self,
                              nusc: NuScenes,
                              sample_token: str,
                              camera_channel: str = 'CAM_FRONT',
                              alpha: float = 0.3,
                              patch_radius: float = 10000,
                              min_polygon_area: float = 1000,
                              render_behind_cam: bool = True,
                              render_outside_im: bool = True,
                              layer_names: List[str] = None,
                              verbose: bool = False,
                              out_path: str = None) -> np.ndarray:
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        near_plane = 1e-8
        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
        if layer_names is None:
            layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

        # # Check layers whether we can render them.
        # for layer_name in layer_names:
        #     assert layer_name in self.map_api.non_geometric_polygon_layers, \
        #         'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert self.map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (self.map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = self.get_records_in_patch(box_coords, layer_names, 'intersect')

        if out_path is not None:
            # Init axes.
            fig = plt.figure(figsize=(9, 16))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, im_size[0])
            ax.set_ylim(0, im_size[1])
            ax.imshow(im)

        points_transform = partial(self.points_transform, poserecord=poserecord, cs_record=cs_record,
                                   cam_intrinsic=cam_intrinsic, near_plane=near_plane, im_size=im_size,
                                   render_behind_cam=render_behind_cam, render_outside_im=render_outside_im)

        # Retrieve and render each record.
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.map_api.non_geometric_line_layers:
                line_list = []
                for token in records_in_patch[layer_name]:
                    record = self.map_api.get(layer_name, token)
                    line = self.map_api.extract_line(record['line_token'])
                    if line.is_empty:  # Skip lines without nodes.
                        continue
                    points = np.array(line.xy)
                    points = points_transform(points)
                    if points is None:
                        continue
                    line = LineString(points)
                    line_list.append(line)
                    # For visualize
                    if out_path is not None:
                        polygon = Polygon(points)
                        ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name],
                                                            alpha=alpha, label=layer_name))
                map_geom.append((layer_name, line_list))
            elif layer_name == 'drivable_area':
                polygon_list = []
                for token in records_in_patch[layer_name]:
                    record = self.map_api.get(layer_name, token)
                    polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in
                                record['polygon_tokens']]
                    for polygon in polygons:
                        ex_points = np.array(polygon.exterior.xy)
                        ex_points = points_transform(ex_points)
                        if ex_points is None:
                            continue
                        interiors = []
                        for interior in polygon.interiors:
                            in_points = np.array(interior.xy)
                            in_points = points_transform(in_points)
                            if in_points is None:
                                continue
                            interiors.append(in_points)
                        polygon = Polygon(ex_points, interiors)
                        polygon = polygon.buffer(0.01)
                        if polygon.geom_type == 'Polygon':
                            polygon = MultiPolygon([polygon])
                        # Filter small polygons
                        if polygon.area < min_polygon_area:
                            continue
                        polygon_list.append(polygon)
                        # For visualize
                        if out_path is not None:
                            ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name],
                                                                alpha=alpha, label=layer_name))
                map_geom.append((layer_name, polygon_list))
            else:
                polygon_list = []
                for token in records_in_patch[layer_name]:
                    record = self.map_api.get(layer_name, token)
                    polygon = self.map_api.extract_polygon(record['polygon_token'])
                    if polygon.is_valid:
                        if not polygon.is_empty:
                            ex_points = np.array(polygon.exterior.xy)
                            ex_points = points_transform(ex_points)
                            if ex_points is None:
                                continue
                            interiors = []
                            for interior in polygon.interiors:
                                in_points = np.array(interior.xy)
                                in_points = points_transform(in_points)
                                if in_points is None:
                                    continue
                                interiors.append(in_points)
                            polygon = Polygon(ex_points, interiors)
                            polygon = polygon.buffer(0.01)
                            if polygon.geom_type == 'Polygon':
                                polygon = MultiPolygon([polygon])
                            # Filter small polygons
                            if polygon.area < min_polygon_area:
                                continue
                            polygon_list.append(polygon)
                            # For visualize
                            if out_path is not None:
                                ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name],
                                                                    alpha=alpha, label=layer_name))
                map_geom.append((layer_name, polygon_list))

        # For visualize
        if out_path is not None:
            # Display the image.
            plt.axis('off')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Convert geometry of each layer into mask and stack them into a numpy tensor.
        # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
        local_box = (im_size[0] // 2, im_size[1] // 2, im_size[1], im_size[0])
        canvas_size = (im_size[1], im_size[0])
        img_mask = self.map_geom_to_mask(map_geom, local_box, canvas_size)
        assert np.all(img_mask.shape[1:] == canvas_size)
        return img_mask

    def render_egoposes_on_fancy_map(self,
                                     nusc: NuScenes,
                                     scene_tokens: List = None,
                                     verbose: bool = True,
                                     out_path: str = None,
                                     render_egoposes: bool = True,
                                     render_egoposes_range: bool = True,
                                     render_legend: bool = True,
                                     bitmap: Optional[BitMap] = None) -> Tuple[np.ndarray, Figure, Axes]:
        """
        Renders each ego pose of a list of scenes on the map (around 40 poses per scene).
        This method is heavily inspired by NuScenes.render_egoposes_on_map(), but uses the map expansion pack maps.
        Note that the maps are constantly evolving, whereas we only released a single snapshot of the data.
        Therefore for some scenes there is a bad fit between ego poses and maps.
        :param nusc: The NuScenes instance to load the ego poses from.
        :param scene_tokens: Optional list of scene tokens corresponding to the current map location.
        :param verbose: Whether to show status messages and progress bar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_egoposes: Whether to render ego poses.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: <np.float32: n, 2>. Returns a matrix with n ego poses in global map coordinates.
        """
        # Settings
        patch_margin = 2
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        scene_blacklist = [499, 515, 517]

        # Get logs by location.
        log_location = self.map_api.map_name
        log_tokens = [log['token'] for log in nusc.log if log['location'] == log_location]
        assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location

        # Filter scenes.
        scene_tokens_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
        if scene_tokens is not None:
            scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]
        assert len(scene_tokens_location) > 0, 'Error: Found 0 valid scenes for location %s!' % log_location

        map_poses = []
        if verbose:
            print('Adding ego poses to map...')
        for scene_token in tqdm(scene_tokens_location, disable=not verbose):
            # Check that the scene is from the correct location.
            scene_record = nusc.get('scene', scene_token)
            scene_name = scene_record['name']
            scene_id = int(scene_name.replace('scene-', ''))
            log_record = nusc.get('log', scene_record['log_token'])
            assert log_record['location'] == log_location, \
                'Error: The provided scene_tokens do not correspond to the provided map location!'

            # Print a warning if the localization is known to be bad.
            if verbose and scene_id in scene_blacklist:
                print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)

            # For each sample in the scene, store the ego pose.
            sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
            for sample_token in sample_tokens:
                sample_record = nusc.get('sample', sample_token)

                # Poses are associated with the sample_data. Here we use the lidar sample_data.
                sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

                # Calculate the pose on the map and append.
                map_poses.append(pose_record['translation'])

        # Check that ego poses aren't empty.
        assert len(map_poses) > 0, 'Error: Found 0 ego poses. Please check the inputs.'

        # Compute number of close ego poses.
        if verbose:
            print('Creating plot...')
        map_poses = np.vstack(map_poses)[:, :2]

        # Render the map patch with the current ego poses.
        min_patch = np.floor(map_poses.min(axis=0) - patch_margin)
        max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
        fig, ax = self.render_map_patch(my_patch, self.map_api.non_geometric_layers, figsize=(10, 10),
                                        render_egoposes_range=render_egoposes_range,
                                        render_legend=render_legend, bitmap=bitmap)

        # Plot in the same axis as the map.
        # Make sure these are plotted "on top".
        if render_egoposes:
            ax.scatter(map_poses[:, 0], map_poses[:, 1], s=20, c='k', alpha=1.0, zorder=2)
        plt.axis('off')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

        return map_poses, fig, ax

    def render_next_roads(self,
                          x: float,
                          y: float,
                          alpha: float = 0.5,
                          figsize: Union[None, float, Tuple[float, float]] = None,
                          bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Renders the possible next roads from a point of interest.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param alpha: The opacity of each layer that gets rendered.
        :param figsize: Size of the whole figure.
        :param bitmap: Optional BitMap object to render below the other map layers.
        """
        # Get next roads.
        next_roads = self.map_api.get_next_roads(x, y)
        layer_names = []
        tokens = []
        for layer_name, layer_tokens in next_roads.items():
            if len(layer_tokens) > 0:
                layer_names.append(layer_name)
                tokens.extend(layer_tokens)

        # Render them.
        fig, ax = self.render_layers(layer_names, alpha, figsize, tokens=tokens, bitmap=bitmap)

        # Render current location with an x.
        ax.plot(x, y, 'x', markersize=12, color='red')

        return fig, ax

    @staticmethod
    def _clip_points_behind_camera(points, near_plane: float):
        """
        Perform clipping on polygons that are partially behind the camera.
        This method is necessary as the projection does not work for points behind the camera.
        Hence we compute the line between the point and the camera and follow that line until we hit the near plane of
        the camera. Then we use that point.
        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param near_plane: If we set the near_plane distance of the camera to 0 then some points will project to
            infinity. Therefore we need to clip these points at the near plane.
        :return: The clipped version of the polygon. This may have fewer points than the original polygon if some lines
            were entirely behind the polygon.
        """
        points_clipped = []
        # Loop through each line on the polygon.
        # For each line where exactly 1 endpoints is behind the camera, move the point along the line until
        # it hits the near plane of the camera (clipping).
        assert points.shape[0] == 3
        point_count = points.shape[1]
        for line_1 in range(point_count):
            line_2 = (line_1 + 1) % point_count
            point_1 = points[:, line_1]
            point_2 = points[:, line_2]
            z_1 = point_1[2]
            z_2 = point_2[2]

            if z_1 >= near_plane and z_2 >= near_plane:
                # Both points are in front.
                # Add both points unless the first is already added.
                if len(points_clipped) == 0 or all(points_clipped[-1] != point_1):
                    points_clipped.append(point_1)
                points_clipped.append(point_2)
            elif z_1 < near_plane and z_2 < near_plane:
                # Both points are in behind.
                # Don't add anything.
                continue
            else:
                # One point is in front, one behind.
                # By convention pointA is behind the camera and pointB in front.
                if z_1 <= z_2:
                    point_a = points[:, line_1]
                    point_b = points[:, line_2]
                else:
                    point_a = points[:, line_2]
                    point_b = points[:, line_1]
                z_a = point_a[2]
                z_b = point_b[2]

                # Clip line along near plane.
                pointdiff = point_b - point_a
                alpha = (near_plane - z_b) / (z_a - z_b)
                clipped = point_a + (1 - alpha) * pointdiff
                assert np.abs(clipped[2] - near_plane) < 1e-6

                # Add the first point (if valid and not duplicate), the clipped point and the second point (if valid).
                if z_1 >= near_plane and (len(points_clipped) == 0 or all(points_clipped[-1] != point_1)):
                    points_clipped.append(point_1)
                points_clipped.append(clipped)
                if z_2 >= near_plane:
                    points_clipped.append(point_2)

        points_clipped = np.array(points_clipped).transpose()
        return points_clipped

    def get_records_in_patch(self,
                             box_coords: Tuple[float, float, float, float],
                             layer_names: List[str] = None,
                             mode: str = 'intersect') -> Dict[str, List[str]]:
        """
        Get all the record token that intersects or within a particular rectangular patch.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: Names of the layers that we want to retrieve in a particular patch.
            By default will always look for all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch,
            "within" will return all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """
        if mode not in ['intersect', 'within']:
            raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        records_in_patch = dict()
        for layer_name in layer_names:
            layer_records = []
            for record in getattr(self.map_api, layer_name):
                token = record['token']
                if self.is_record_in_patch(layer_name, token, box_coords, mode):
                    layer_records.append(token)

            records_in_patch.update({layer_name: layer_records})

        return records_in_patch

    def is_record_in_patch(self,
                           layer_name: str,
                           token: str,
                           box_coords: Tuple[float, float, float, float],
                           mode: str = 'intersect') -> bool:
        """
        Query whether a particular record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular record intersects or is within a particular patch.
        """
        if mode not in ['intersect', 'within']:
            raise ValueError("Mode {} is not valid, choice=('intersect', 'within')".format(mode))

        if layer_name in self.map_api.lookup_polygon_layers:
            return self._is_polygon_record_in_patch(token, layer_name, box_coords, mode)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._is_line_record_in_patch(token, layer_name, box_coords,  mode)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def layers_on_point(self, x: float, y: float, layer_names: List[str] = None) -> Dict[str, str]:
        """
        Returns all the polygonal layers that a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_names: The names of the layers to search for.
        :return: All the polygonal layers that a particular point is on.
        """
        # Default option.
        if layer_names is None:
            layer_names = self.map_api.non_geometric_polygon_layers

        layers_on_point = dict()
        for layer_name in layer_names:
            layers_on_point.update({layer_name: self.record_on_point(x, y, layer_name)})

        return layers_on_point

    def record_on_point(self, x: float, y: float, layer_name: str) -> str:
        """
        Query what record of a layer a particular point is on.
        :param x: x coordinate of the point of interest.
        :param y: y coordinate of the point of interest.
        :param layer_name: The non geometric polygonal layer name that we are interested in.
        :return: The first token of a layer a particular point is on or '' if no layer is found.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError("{} is not a polygon layer".format(layer_name))

        point = Point(x, y)
        records = getattr(self.map_api, layer_name)

        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                for polygon in polygons:
                    if point.within(polygon):
                        return record['token']
                    else:
                        pass
        else:
            for record in records:
                polygon = self.map_api.extract_polygon(record['polygon_token'])
                if point.within(polygon):
                    return record['token']
                else:
                    pass

        # If nothing is found, return an empty string.
        return ''

    def extract_polygon(self, polygon_token: str) -> Polygon:
        """
        Construct a shapely Polygon object out of a polygon token.
        :param polygon_token: The token of the polygon record.
        :return: The polygon wrapped in a shapely Polygon object.
        """
        polygon_record = self.map_api.get('polygon', polygon_token)

        exterior_coords = [(self.map_api.get('node', token)['x'], self.map_api.get('node', token)['y'])
                           for token in polygon_record['exterior_node_tokens']]

        interiors = []
        for hole in polygon_record['holes']:
            interior_coords = [(self.map_api.get('node', token)['x'], self.map_api.get('node', token)['y'])
                               for token in hole['node_tokens']]
            if len(interior_coords) > 0:  # Add only non-empty holes.
                interiors.append(interior_coords)

        return Polygon(exterior_coords, interiors)

    def extract_line(self, line_token: str) -> LineString:
        """
        Construct a shapely LineString object out of a line token.
        :param line_token: The token of the line record.
        :return: The line wrapped in a LineString object.
        """
        line_record = self.map_api.get('line', line_token)
        line_nodes = [(self.map_api.get('node', token)['x'], self.map_api.get('node', token)['y'])
                      for token in line_record['node_tokens']]

        return LineString(line_nodes)

    def get_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the geometric object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of the line representation.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            return self._get_polygon_bounds(layer_name, token)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._get_line_bounds(layer_name, token)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _get_polygon_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the extremities of the polygon object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the polygon or polygons (for drivable_area) representation.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError("{} is not a record with polygon representation".format(token))

        record = self.map_api.get(layer_name, token)

        if layer_name == 'drivable_area':
            polygons = [self.map_api.get('polygon', polygon_token) for polygon_token in record['polygon_tokens']]
            exterior_node_coords = []

            for polygon in polygons:
                nodes = [self.map_api.get('node', node_token) for node_token in polygon['exterior_node_tokens']]
                node_coords = [(node['x'], node['y']) for node in nodes]
                exterior_node_coords.extend(node_coords)

            exterior_node_coords = np.array(exterior_node_coords)
        else:
            exterior_nodes = [self.map_api.get('node', token) for token in record['exterior_node_tokens']]
            exterior_node_coords = np.array([(node['x'], node['y']) for node in exterior_nodes])

        xs = exterior_node_coords[:, 0]
        ys = exterior_node_coords[:, 1]

        x2 = xs.max()
        x1 = xs.min()
        y2 = ys.max()
        y1 = ys.min()

        return x1, y1, x2, y2

    def _get_line_bounds(self, layer_name: str, token: str) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the line object that corresponds to a non geometric record.
        :param layer_name: Name of the layer that we are interested in.
        :param token: Token of the record.
        :return: min_x, min_y, max_x, max_y of of the line representation.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a record with line representation".format(token))

        record = self.map_api.get(layer_name, token)
        nodes = [self.map_api.get('node', node_token) for node_token in record['node_tokens']]
        node_coords = [(node['x'], node['y']) for node in nodes]
        node_coords = np.array(node_coords)

        xs = node_coords[:, 0]
        ys = node_coords[:, 1]

        x2 = xs.max()
        x1 = xs.min()
        y2 = ys.max()
        y1 = ys.min()

        return x1, y1, x2, y2

    def _is_polygon_record_in_patch(self,
                                    token: str,
                                    layer_name: str,
                                    box_coords: Tuple[float, float, float, float],
                                    mode: str = 'intersect') -> bool:
        """
        Query whether a particular polygon record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular polygon record intersects or is within a particular patch.
        """
        if layer_name not in self.map_api.lookup_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        x_min, y_min, x_max, y_max = box_coords
        record = self.map_api.get(layer_name, token)
        rectangular_patch = box(x_min, y_min, x_max, y_max)

        if layer_name == 'drivable_area':
            polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
            geom = MultiPolygon(polygons)
        else:
            geom = self.map_api.extract_polygon(record['polygon_token'])

        if mode == 'intersect':
            return geom.intersects(rectangular_patch)
        elif mode == 'within':
            return geom.within(rectangular_patch)

    def _is_line_record_in_patch(self,
                                 token: str,
                                 layer_name: str,
                                 box_coords: Tuple[float, float, float, float],
                                 mode: str = 'intersect') -> bool:
        """
        Query whether a particular line record is in a rectangular patch.
        :param layer_name: The layer name of the record.
        :param token: The record token.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param mode: "intersect" means it will return True if the geometric object intersects the patch and False
        otherwise, "within" will return True if the geometric object is within the patch and False otherwise.
        :return: Boolean value on whether a particular line  record intersects or is within a particular patch.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        # Retrieve nodes of this line.
        record = self.map_api.get(layer_name, token)
        node_recs = [self.map_api.get('node', node_token) for node_token in record['node_tokens']]
        node_coords = [[node['x'], node['y']] for node in node_recs]
        node_coords = np.array(node_coords)

        # A few lines in Queenstown have zero nodes. In this case we return False.
        if len(node_coords) == 0:
            return False

        # Check that nodes fall inside the path.
        x_min, y_min, x_max, y_max = box_coords
        cond_x = np.logical_and(node_coords[:, 0] < x_max, node_coords[:, 0] > x_min)
        cond_y = np.logical_and(node_coords[:, 1] < y_max, node_coords[:, 1] > y_min)
        cond = np.logical_and(cond_x, cond_y)
        if mode == 'intersect':
            return np.any(cond)
        elif mode == 'within':
            return np.all(cond)

    def _render_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            self._render_polygon_layer(ax, layer_name, alpha, tokens)
        elif layer_name in self.map_api.non_geometric_line_layers:
            self._render_line_layer(ax, layer_name, alpha, tokens)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _render_polygon_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Renders an individual non-geometric polygon layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        first_time = True
        records = getattr(self.map_api, layer_name)
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    if first_time:
                        label = layer_name
                        first_time = False
                    else:
                        label = None
                    ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label))
        else:
            for record in records:
                polygon = self.map_api.extract_polygon(record['polygon_token'])

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None

                ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                    label=label))

    def _render_line_layer(self, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Renders an individual non-geometric line layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        first_time = True
        records = getattr(self.map_api, layer_name)
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        for record in records:
            if first_time:
                label = layer_name
                first_time = False
            else:
                label = None
            line = self.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            if layer_name == 'traffic_light':
                # Draws an arrow with the physical traffic light as the starting point, pointing to the direction on
                # where the traffic light points.
                ax.add_patch(Arrow(xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], color=self.color_map[layer_name],
                                   label=label))
            else:
                ax.plot(xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label)

    def _get_layer_geom(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str) -> List[Geometry]:
        """
        Wrapper method that gets the geometries for each layer.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_name: Name of map layer to be converted to binary map mask patch.
        :return: List of geometries for the given layer.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            return self._get_layer_polygon(patch_box, patch_angle, layer_name)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._get_layer_line(patch_box, patch_angle, layer_name)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _layer_geom_to_mask(self,
                            layer_name: str,
                            layer_geom: List[Geometry],
                            local_box: Tuple[float, float, float, float],
                            canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Wrapper method that gets the mask for each layer's geometries.
        :param layer_name: The name of the layer for which we get the masks.
        :param layer_geom: List of the geometries of the layer specified in layer_name.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            return self._polygon_geom_to_mask(layer_geom, local_box, layer_name, canvas_size)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._line_geom_to_mask(layer_geom, local_box, layer_name, canvas_size)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    @staticmethod
    def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
        """
        Convert a polygon or multipolygon list to an image mask ndarray.
        :param polygons: List of Shapely polygons to be converted to numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray polygon mask.
        """
        if not polygons:
            return mask

        def int_coords(x):
            # function to round and convert to int
            return np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        cv2.fillPoly(mask, interiors, 0)
        return mask

    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        if lines.geom_type == 'MultiLineString':
            for line in lines:
                coords = np.asarray(list(line.coords), np.int32)
                coords = coords.reshape((-1, 2))
                cv2.polylines(mask, [coords], False, 1, 2)
        else:
            coords = np.asarray(list(lines.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)

        return mask

    def _polygon_geom_to_mask(self,
                              layer_geom: List[Polygon],
                              local_box: Tuple[float, float, float, float],
                              layer_name: str,
                              canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert polygon inside patch to binary mask and return the map patch.
        :param layer_geom: list of polygons for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch with the size canvas_size.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]

        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        for polygon in layer_geom:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.affine_transform(new_polygon,
                                                        [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_polygon = affinity.scale(new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])

                # if new_polygon.area < 1000:
                #     continue

                if not isinstance(new_polygon, MultiPolygon):
                    print(new_polygon)
                    
                    continue

                map_mask = self.mask_for_polygons(new_polygon, map_mask)

        return map_mask

    def _line_geom_to_mask(self,
                           layer_geom: List[LineString],
                           local_box: Tuple[float, float, float, float],
                           layer_name: str,
                           canvas_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Convert line inside patch to binary mask and return the map patch.
        :param layer_geom: list of LineStrings for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch in a canvas size.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h/patch_h
        scale_width = canvas_w/patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        if layer_name == 'traffic_light':
            return None

        for line in layer_geom:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                map_mask = self.mask_for_lines(new_line, map_mask)
        return map_mask

    def _get_layer_polygon(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str) -> List[Polygon]:
        """
         Retrieve the polygons of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: List of Polygon in a patch box.
         """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def _get_layer_line(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str) -> Optional[List[LineString]]:
        """
        Retrieve the lines of a particular layer within the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :return: List of LineString in a patch box.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_api, layer_name)
        for record in records:
            line = self.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle,
                                           origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    @staticmethod
    def get_patch_coord(patch_box: Tuple[float, float, float, float],
                        patch_angle: float = 0.0) -> Polygon:
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch

    def _get_figsize(self, figsize: Union[None, float, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Utility function that scales the figure size by the map canvas size.
        If figsize is:
        - None      => Return default scale.
        - Scalar    => Scale canvas size.
        - Two-tuple => Use the specified figure size.
        :param figsize: The input figure size.
        :return: The output figure size.
        """
        # Divide canvas size by arbitrary scalar to get into cm range.
        canvas_size = np.array(self.map_api.canvas_edge)[::-1] / 200

        if figsize is None:
            return tuple(canvas_size)
        elif type(figsize) in [int, float]:
            return tuple(canvas_size * figsize)
        elif type(figsize) == tuple and len(figsize) == 2:
            return figsize
        else:
            raise Exception('Error: Invalid figsize: %s' % figsize)
