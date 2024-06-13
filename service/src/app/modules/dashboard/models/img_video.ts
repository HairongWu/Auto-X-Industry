export interface ImgVideo {
  id: number;
  caption: string;
  detected_objects?: number;
  alarms: number;
  creator?: string;
  avatar?: string;
  recognized_objects?: number;
  ending_in?: string;
  image: string;
}
