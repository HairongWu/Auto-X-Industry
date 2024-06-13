import { Component, Input, OnInit } from '@angular/core';
import { DI } from '../../../models/di';
import { NgStyle, CurrencyPipe } from '@angular/common';

@Component({
    selector: '[di-dual-card]',
    templateUrl: './di-dual-card.component.html',
    standalone: true,
    imports: [NgStyle, CurrencyPipe],
})
export class DIDualCardComponent implements OnInit {
  @Input() nft: DI = <DI>{};

  constructor() {}

  ngOnInit(): void {}
}
