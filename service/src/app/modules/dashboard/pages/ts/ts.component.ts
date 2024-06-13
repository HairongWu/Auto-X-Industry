import { Component, OnInit } from '@angular/core';
import { TS } from '../../models/ts';
import { TSChartCardComponent } from '../../components/ts/ts-chart-card/ts-chart-card.component';
import { TSHeaderComponent } from '../../components/ts/ts-header/ts-header.component';

@Component({
    selector: 'app-ts',
    templateUrl: './ts.component.html',
    standalone: true,
    imports: [
        TSHeaderComponent,
        TSChartCardComponent,
    ],
})
export class TSComponent implements OnInit {
  nft: Array<TS>;
  
  constructor() {
    this.nft = [
      {
        id: 34356771,
        category: 'Etherium rate',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 0,
        ending_in: '20s',
        recognized_objects: 1,
        image: './assets/images/img-01.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356772,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 0,
        ending_in: '40s',
        recognized_objects: 2,
        image: './assets/images/img-02.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356773,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 5,
        ending_in: '30s',
        recognized_objects: 5,
        image: './assets/images/img-03.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356774,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 0,
        ending_in: '100s',
        recognized_objects: 49,
        image: './assets/images/img-04.jpg',
        avatar: './assets/avatars/auto-x.png',
      },      
      {
        id: 34356771,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 0,
        ending_in: '20s',
        recognized_objects: 1,
        image: './assets/images/img-01.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356772,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 0,
        ending_in: '40s',
        recognized_objects: 2,
        image: './assets/images/img-02.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356773,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
        alarms: 5,
        ending_in: '30s',
        recognized_objects: 5,
        image: './assets/images/img-03.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 34356774,
        category: 'sensor data',
        creator: 'Auto-X AI Server',
        next_value: 3274.94,
        data: [2100, 3200, 3200, 2400, 2400, 1800, 1800, 2400, 2400, 3200, 3200, 3000, 3000, 3250, 3250],
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
