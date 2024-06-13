import { Component, OnInit } from '@angular/core';
import { ImgVideo } from '../../models/img_video';
import { ImgVideoDualCardComponent } from '../../components/img_video/img_video-dual-card/img_video-dual-card.component';
import { ImgVideoHeaderComponent } from '../../components/img_video/img_video-header/img_video-header.component';

@Component({
    selector: 'app-img_video',
    templateUrl: './img_video.component.html',
    standalone: true,
    imports: [
      ImgVideoHeaderComponent,
      ImgVideoDualCardComponent
    ],
})
export class ImgVideoComponent implements OnInit {
  nft: Array<ImgVideo>;

  constructor() {
    this.nft = [
      {
        id: 34356771,
        caption: 'Surface Cracks in Concrete Structures',
        creator: 'Auto-X AI Server',
        detected_objects: 1,
        alarms: 0,
        ending_in: '20s',
        recognized_objects: 1,
        image: './assets/images/img-01.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356772,
        caption: 'A man drving a green car',
        creator: 'Auto-X AI Server',
        detected_objects: 2,
        alarms: 0,
        ending_in: '40s',
        recognized_objects: 2,
        image: './assets/images/img-02.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356773,
        caption: 'A route with cars and traffics',
        creator: 'Auto-X AI Server',
        detected_objects: 10,
        alarms: 5,
        ending_in: '30s',
        recognized_objects: 5,
        image: './assets/images/img-03.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356774,
        caption: 'A rack of goods in a supermarket',
        creator: 'Auto-X AI Server',
        detected_objects: 50,
        alarms: 0,
        ending_in: '100s',
        recognized_objects: 49,
        image: './assets/images/img-04.jpg',
        avatar: './assets/avatars/auto-x.png',
      },      
    ];
  }

  ngOnInit(): void {}
}
