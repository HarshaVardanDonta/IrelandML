from dagster import Definitions, load_assets_from_modules

from . import assets, structured_asset1, structured_asset2


all_assets = load_assets_from_modules([ structured_asset2, assets, structured_asset1])

defs = Definitions(
    assets=all_assets,
)
