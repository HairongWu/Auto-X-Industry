import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TSHeaderComponent } from './ts-header.component';

describe('TSHeaderComponent', () => {
  let component: TSHeaderComponent;
  let fixture: ComponentFixture<TSHeaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [TSHeaderComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TSHeaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
