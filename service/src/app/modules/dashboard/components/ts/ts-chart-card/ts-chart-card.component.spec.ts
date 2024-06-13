import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TSChartCardComponent } from './ts-chart-card.component';

describe('TSChartCardComponent', () => {
  let component: TSChartCardComponent;
  let fixture: ComponentFixture<TSChartCardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [TSChartCardComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TSChartCardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
