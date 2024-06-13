import { Component, Input, OnInit } from '@angular/core';
import { ImgVideo } from '../../../models/img_video';
import { NgStyle, CurrencyPipe } from '@angular/common';

@Component({
    selector: '[img_video-dual-card]',
    templateUrl: './img_video-dual-card.component.html',
    standalone: true,
    imports: [NgStyle, CurrencyPipe],
})
export class ImgVideoDualCardComponent implements OnInit {
  @Input() nft: ImgVideo = <ImgVideo>{};

  constructor() {}

  ngOnInit(): void {}
}
