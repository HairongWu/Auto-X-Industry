import { Component, OnInit } from '@angular/core';
import { DI } from '../../models/di';
import { DIDualCardComponent } from '../../components/di/di-dual-card/di-dual-card.component';
import { DIHeaderComponent } from '../../components/di/di-header/di-header.component';

@Component({
    selector: 'app-di',
    templateUrl: './di.component.html',
    standalone: true,
    imports: [
      DIHeaderComponent,
      DIDualCardComponent,
    ],
})
export class DIComponent implements OnInit {
  nft: Array<DI>;

  constructor() {
    this.nft = [
      {
        id: 54356771,
        category: 'invoice',
        creator: 'Auto-X AI Server',
        pages: 1,
        alarms: 0,
        ending_in: '20s',
        tables: 1,
        image: './assets/images/di_01.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 54356772,
        category: 'receipt',
        creator: 'Auto-X AI Server',
        pages: 1,
        alarms: 0,
        ending_in: '40s',
        tables: 1,
        image: './assets/images/di_02.png',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 54356773,
        category: 'questionnaire',
        creator: 'Auto-X AI Server',
        pages: 2,
        alarms: 2,
        ending_in: '30s',
        tables: 1,
        image: './assets/images/di_03.jpg',
        avatar: './assets/avatars/auto-x.png',
      },
      {
        id: 54356774,
        category: 'form',
        creator: 'Auto-X AI Server',
        pages: 1,
        alarms: 0,
        ending_in: '100s',
        tables: 2,
        image: './assets/images/di_04.jpg',
        avatar: './assets/avatars/auto-x.png',
      },      
    ];
  }

  ngOnInit(): void {}
}
