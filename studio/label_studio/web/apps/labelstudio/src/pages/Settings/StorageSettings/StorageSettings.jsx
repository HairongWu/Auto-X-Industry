import React from "react";
import { Columns } from "../../../components/Columns/Columns";
import { Description } from "../../../components/Description/Description";
import { Block, cn } from "../../../utils/bem";
import { StorageSet } from "./StorageSet";
import { GenerationSet } from "./GenerationSet";
import { ScrapingSet } from "./ScrapingSet";
import "./StorageSettings.styl";
import { isInLicense, LF_CLOUD_STORAGE_FOR_MANAGERS } from "../../../utils/license-flags";

const isAllowCloudStorage = !isInLicense(LF_CLOUD_STORAGE_FOR_MANAGERS);

export const StorageSettings = () => {
  const rootClass = cn("storage-settings");

  return isAllowCloudStorage ? (
    <Block name="storage-settings">
      <Description style={{ marginTop: 0 }}>
        Use cloud or database storage as the source for your labeling tasks, the target of your completed annotations or the models used in your project.
      </Description>

      <Columns count={2} gap="40px" size="320px" className={rootClass}>
        <StorageSet title="Source Storage" buttonLabel="Add Source Storage" rootClass={rootClass} />
        <GenerationSet title="Dataset Generation" buttonLabel="Generate from Models" rootClass={rootClass} />
        <ScrapingSet title="Dataset Scraping" buttonLabel="Scrape for Websites" rootClass={rootClass} />
      </Columns>
    </Block>
  ) : null;
};

StorageSettings.title = "Data Source";
StorageSettings.path = "/storage";
